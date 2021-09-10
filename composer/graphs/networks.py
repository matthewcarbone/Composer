from pathlib import Path
import pickle
import time

from dgllife.model import MPNNPredictor
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def collating_function_graph_to_vector(batch):
    """Collates the graph-fixed length vector combination. Recall that
    in this case, each element of batch is a three vector containing the
    graph, the target and the ID."""

    qm9ids = torch.tensor([xx[2] for xx in batch]).long()

    # Each target is the same length, so we can use standard batching for
    # it.
    targets = torch.tensor([xx[1] for xx in batch])

    # However, graphs are not of the same "length" (diagonally on the
    # adjacency matrix), so we need to be careful. Usually, dgl's batch
    # method would work just fine here, but for multi-gpu training, we
    # need to catch some subtleties, since the batch itself is split apart
    # equally onto many GPU's, but torch doesn't know how to properly split
    # a batch of graphs. So, we manually partition the graphs here, and
    # will batch the output of the collating function before training.
    # This is now just a list of graphs.
    graphs = [xx[0] for xx in batch]

    return (graphs, targets, qm9ids)


class LightningMPNN(pl.LightningModule):

    def __init__(
        self,
        data_dir,
        n_node_features,
        n_edge_features,
        output_size=200,
        hidden_node_size=64,
        hidden_edge_size=128,
        criterion='mse',
        last_activation='softplus',
        train_batch_size=64,
        val_batch_size=64,
        plateau_scheduler_patience=100,
        plateau_scheduler_min_lr=1e-7,
        plateau_scheduler_factor=0.95,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MPNNPredictor(
            node_in_feats=n_node_features,
            edge_in_feats=n_edge_features,
            node_out_feats=hidden_node_size,
            edge_hidden_feats=hidden_edge_size,
            n_tasks=output_size
        )

        if self.hparams.criterion == 'mse':
            self._criterion = nn.MSELoss(reduction='mean')
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        if self.hparams.last_activation == 'softplus':
            self._last_activation = nn.Softplus()
        elif self.hparams.last_activation == 'sigmoid':
            self._last_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown last activation {last_activation}")

        # Custom
        self._print_every_epoch = 20
        self._epoch_dt = None
        self._data_dir = data_dir

    def set_non_hyperparameters(self, workers=4, adam_kwargs={'lr': 1e-2}):
        self._workers = workers
        self._adam_kwargs = adam_kwargs

    def train_dataloader(self):
        path = Path(self._data_dir) / "train.pkl"
        print(f"Loading training data from {path}")
        ds = pickle.load(open(path, "rb"))
        return DataLoader(
            ds, batch_size=self.hparams.train_batch_size,
            num_workers=self._workers, persistent_workers=True,
            pin_memory=True, shuffle=True
        )

    def val_dataloader(self):
        path = Path(self._data_dir) / "val.pkl"
        print(f"Loading validation data from {path}")
        ds = pickle.load(open(path, "rb"))
        return DataLoader(
            ds, batch_size=self.hparams.val_batch_size,
            num_workers=self._workers, persistent_workers=True,
            pin_memory=True, shuffle=False
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self._adam_kwargs)
        print("Optimizers configured")
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams.plateau_scheduler_patience,
                    min_lr=self.hparams.plateau_scheduler_min_lr,
                    factor=self.hparams.plateau_scheduler_factor
                ),
                'monitor': 'val_loss'
            }
        }

    def get_batch(batch):
        """Parses a batch from the Loaders to the model-compatible features.
        
        Parameters
        ----------
        batch : TYPE
            Description
        """


    def forward(self, x):
        """In Torch Lightning, the forward method defines the prediction/
        inference action."""

        return self.model(x)

    def _single_forward_step(self, batch, batch_index):
        """Summary
        
        Parameters
        ----------
        batch : TYPE
            Description
        batch_index : TYPE
            Description
        """

        (i)
        pred = self(x)



