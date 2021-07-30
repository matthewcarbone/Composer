import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# X_train, mu_train, std_train = generate_data(n_samples=2**12 * 5)
# X_val, mu_val, std_val = generate_data(n_samples=2**6 * 5)


def reparameterize(mu, log_var):
    """Implements the 'reparameterization trick'."""

    std = torch.exp(0.5 * log_var)  # standard deviation
    eps = torch.randn_like(std)     # `randn_like` as we need the same size
    sample = mu + eps * std         # sampling as if coming from input space
    return sample


def vae_kl_loss(mu, logvar):
    """ KL-Divergence of VAE ouputted (mu, logvar) with unit Gaussian"""

    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())


def split_latents(z):
    """Takes a latent space vector and separates it into the shift and the
    rest of the latent variables."""

    return z[:, 0], z[:, 1:]


class NetBlock(nn.Module):

    def __init__(self, n1, n2, batch_norm=True, activation=nn.ReLU()):
        super().__init__()

        block = [nn.Linear(n1, n2)]
        if batch_norm:
            block.append(nn.BatchNorm1d(n2))
        block.append(activation)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class Net(nn.Module):

    def __init__(
        self, input_size, architecture, output_size,
        last_activation=nn.Softplus()
    ):
        super().__init__()

        # Initialize the net
        net = [nn.Linear(input_size, architecture[0])] + [
            NetBlock(architecture[ii], architecture[ii + 1])
            for ii in range(len(architecture) - 1)
        ] + [nn.Linear(architecture[-1], output_size)]
        if last_activation is not None:
            net.append(last_activation)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class DecoderNet(nn.Module):

    def __init__(
        self, n_latent_not_dx, decoder_architecture, hidden=16, grid_size=100,
        last_activation=nn.Softplus()
    ):
        """
        Parameters
        ----------
        n_latent_not_dx : int
            The number of variables in the latent space that do *not*
            correspond to the invariant neuron.
        decoder_architecture : list[int]
            A list of integers corresponding to the architecture of the latent
            space part of the decoder network.
        hidden : int
            The number of hidden states contained in this simple network. Both
            the invariant and relative behavior neurons are mapped to this
            size. Default is 128.
        grid_size : int
            The size of the coordinate grid. Default is 100.
        last_activation
            Activation function from torch.nn which is the last non-linear
            transform to be applied to the data. Default is Softplus
            (to ensure output is positive).
        """

        super().__init__()

        self.n_latent_not_dx = n_latent_not_dx
        self.hidden = hidden
        self.grid_size = grid_size

        # Define mappings on the x-coordinates
        self.coord_linear_1 = nn.Linear(1, self.hidden)

        # Define mappings on the latents
        self.latent_linear_1 = nn.Linear(self.n_latent_not_dx, self.hidden)

        # Final network
        self.final_net = Net(
            self.hidden, decoder_architecture, 1,
            last_activation=last_activation
        )

    def forward(self, coord, z):
        """Executes the forward pass for the Decoder network.

        Parameters
        ----------
        coord : torch.Tensor
            The coordinate tensor of shape (batch size, grid size). Note this
            is usually just a copied grid, where each row has been modified
            slightly by some dx.
        z : torch.Tensor
            The latent space tensor of shape (batch size, n_latent_not_dx).

        Returns
        -------
        torch.Tensor
        """

        # Hidden layers on the coordinates
        coord = self.coord_linear_1(coord.reshape(-1, 1))
        coord = coord.reshape(-1, self.grid_size, self.hidden)

        # Hidden layers on the latent space
        z = self.latent_linear_1(z).reshape(-1, 1, self.hidden)

        # Broadcasting; combine the results
        # Resulting shape is (BS, grid_size, hidden)
        res = coord + z

        # Apply the "standard" decoder network. The input to the decoder net
        # will be (BS * grid_size, hidden) and should map to
        # (BS * grid_size, 1). That will finally be reshaped back to
        # (BS, grid_size).
        res = self.final_net(res.reshape(-1, self.hidden))

        # Final shape is (BS * grid_size, 1); reshape for final output
        return res.reshape(-1, self.grid_size)


class Model(pl.LightningModule):

    def __init__(
        self,
        data,
        train_batch_size=2048,
        valid_batch_size=1024,
        print_every_epoch=1,
        latent_space_size=2,
        grid_size=128,
        kl_lambda=0.0,
        kl_ramp_epochs=None,
        architecture=[64, 32],
        dx_prior=0.1,
        decoder_hidden=16,
        criterion=nn.MSELoss(reduction='mean'),
        last_activation=nn.Softplus(),
        workers=3
    ):
        super().__init__()

        assert latent_space_size > 0

        print(
            "Model initializing with TI-VAE; "
            f"Latent size={latent_space_size} (excludes dx)"
        )
        print(f"dx_prior is {dx_prior:.02e}")

        # Initialize the encoder
        self.encoder = Net(
            grid_size, architecture, 2 * latent_space_size + 1,
            last_activation=None
        )

        # Initialize the decoder
        self.decoder = DecoderNet(
            latent_space_size, list(reversed(architecture)),
            hidden=decoder_hidden, grid_size=grid_size,
            last_activation=last_activation
        )

        self.criterion = criterion

        # Custom
        self._X_train = data['train']
        self._X_val = data['val']
        self._latent_space_size = latent_space_size
        self._print_every_epoch = print_every_epoch
        self._epoch_dt = 0.0
        self._kl_lambda = kl_lambda
        self._kl_ramp_epochs = kl_ramp_epochs
        if self._kl_ramp_epochs is not None:
            self._kl_ramp_strength = 0.0
        else:
            self._kl_ramp_strength = 1.0
        self._dx_prior = dx_prior  # Control the overall strength of the shift
        self._train_batch_size = train_batch_size
        self._valid_batch_size = valid_batch_size
        self._workers = workers

    def _split_encoder_output(self, x):
        x = x.reshape(-1, 2, self._latent_space_size)

        # Use the first set as the distributions
        mu = x[:, 0, :]

        # The second set is the log_variances
        log_var = x[:, 1, :]

        return mu, log_var

    def forward(self, x, training=True):
        # in lightning, forward defines the prediction/inference actions

        embedding = self.encoder(x)

        # The first embedding is dx; the rest are the true latent variables.
        # This will disconnect dx from the KL divergence part of the backprop
        dx = embedding[:, 0]
        mu, log_var = self._split_encoder_output(embedding[:, 1:])

        if training:
            return dx, reparameterize(mu, log_var), mu, log_var

        # If not training, return just dx, the mean and standard deviation
        return dx, mu, torch.exp(0.5 * log_var)

    def _single_forward_step(self, batch, batch_index):
        """Executes a single forward pass given some batch and batch index.
        In this model, we first encode using self(x)"""

        x, = batch
        dx, z, mu, log_var = self(x)

        # Create the grid
        # Use the latent space information to shift the grid
        # Since it was initialized here, I think I have to manually
        # move it to the same device as x.
        _grid = torch.linspace(
            -1.0, 1.0, x.shape[1], device=self.device
        ).expand(*x.shape)

        # Execute the shift and decode
        _grid = _grid + self._dx_prior * dx.reshape(-1, 1)
        x_hat = self.decoder(_grid, z)

        # Compute losses
        mse_loss = self.criterion(x_hat, x)  # reduction = mean already applies
        kl_loss = vae_kl_loss(mu, log_var) / x.shape[0]

        loss = self._kl_lambda * self._kl_ramp_strength * kl_loss + mse_loss

        # "loss" key is required; backprop is run on this object
        return {
            'loss': loss,
            'mse_loss': mse_loss.item(),
            'kl_loss': kl_loss.item()
        }

    def _log_outputs(
        self, outputs, what='train', keys=['loss', 'mse_loss', 'kl_loss']
    ):
        d = {}
        for key in keys:
            tmp_loss = torch.tensor([x[key] for x in outputs]).mean().item()
            d[key] = tmp_loss
            self.log(f"{what}_{key}", tmp_loss, on_step=False, on_epoch=True)
        return d

    def training_step(self, batch, batch_idx):
        return self._single_forward_step(batch, batch_idx)

    def on_train_epoch_start(self):
        self._epoch_dt = time.time()

    def training_epoch_end(self, outputs):
        d = self._log_outputs(outputs, 'train')
        epoch = self.trainer.current_epoch + 1
        dt = (time.time() - self._epoch_dt) / 60.0
        if epoch % self._print_every_epoch == 0:
            loss = d['loss']
            print(f"Epoch {epoch:05}: dt: {dt:.02} m; train loss: {loss:.03e}")

        # Ramp the kl_loss if necessary
        if self._kl_ramp_epochs is not None:
            self._kl_ramp_strength = \
                np.tanh(2.0 * epoch / self._kl_ramp_epochs)

    def validation_step(self, batch, batch_idx):
        return self._single_forward_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        d = self._log_outputs(outputs, 'val')
        epoch = self.trainer.current_epoch + 1

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=False, on_epoch=True)
        self.log('kl_lambda', self._kl_lambda, on_step=False, on_epoch=True)
        self.log(
            'kl_ramp_strength', self._kl_ramp_strength,
            on_step=False, on_epoch=True
        )

        if epoch % self._print_every_epoch == 0:
            loss = d['loss']
            print(f"Epoch {epoch:05}: val loss {loss:.03e}; lr {lr:.03e}")

    def configure_optimizers(self):
        print("configure_optimizers called")
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=10, min_lr=1e-7, factor=0.95
                ),
                'monitor': 'val_loss'
            }
        }

    def train_dataloader(self):
        print("train_dataloader called")
        ds = TensorDataset(self._X_train)
        return DataLoader(
            ds, batch_size=self._train_batch_size, num_workers=self._workers,
            persistent_workers=True, pin_memory=True
        )

    def val_dataloader(self):
        print("val_dataloader called")
        ds = TensorDataset(self._X_val)
        return DataLoader(
            ds, batch_size=self._valid_batch_size, num_workers=self._workers,
            persistent_workers=True, pin_memory=True
        )

    def on_train_end(self):
        """Logs information as training ends."""

        if self.trainer.global_rank == 0:
            epoch = self.trainer.current_epoch + 1
            if epoch < self.trainer.max_epochs:
                print(
                    "Early stopping criteria reached at "
                    f"epoch {epoch}/{self.trainer.max_epochs}"
                )

# class CustomTrainer(pl.Trainer):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def export_csv_log(self):
#         path = Path(self.logger.experiment.log_dir) / Path("custom_metrics.csv")
#         t = pd.DataFrame([d for d in self.logger.experiment.metrics if 'train_loss' in d])
#         v = pd.DataFrame([d for d in self.logger.experiment.metrics if 'val_loss' in d])
#         df = pd.concat([t, v], join='outer', axis=1)
#         df = df.loc[:,~df.columns.duplicated()]
#         df = df[[
#             'epoch', 'train_loss', 'train_mse_loss', 'train_kl_loss',
#             'val_loss', 'val_mse_loss', 'val_kl_loss', 'lr', 'kl_lambda',
#             'kl_ramp_strength'
#         ]]
#         df.to_csv(path, index=False)
