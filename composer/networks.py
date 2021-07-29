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


class NetBlock(nn.Module):

    def __init__(self, n1, n2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(n1, n2),
            nn.BatchNorm1d(n2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class Model(pl.LightningModule):

    def __init__(
        self, data, print_every_epoch=20, latent_space_size=2, kl_lambda=0.0,
        kl_ramp_epochs=None, architecture=[128, 64, 32]
    ):
        super().__init__()

        # Initialize the encoder
        encoder = [
            NetBlock(architecture[ii], architecture[ii + 1])
            for ii in range(len(architecture) - 1)
        ] + [nn.Linear(architecture[-1], 2 * latent_space_size)]
        self.encoder = nn.Sequential(*encoder)

        # Initialize the decoder
        r_architecture = list(reversed(architecture))
        decoder = [nn.Linear(latent_space_size, r_architecture[0])] + [
            NetBlock(r_architecture[ii], r_architecture[ii + 1])
            for ii in range(len(r_architecture) - 1)
        ] + [nn.Linear(r_architecture[-1], r_architecture[-1]), nn.Softplus()]

        self.decoder = nn.Sequential(*decoder)

        self.mse_loss = nn.MSELoss(reduction='mean')

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

    def _split_encoder_output(self, x):
        x = x.view(-1, 2, self._latent_space_size)

        # Use the first set as the distributions
        mu = x[:, 0, :]

        # The second set is the log_variances
        log_var = x[:, 1, :]

        return mu, log_var

    def forward(self, x, training=True):
        # in lightning, forward defines the prediction/inference actions

        embedding = self.encoder(x)
        mu, log_var = self._split_encoder_output(embedding)

        if training:
            return reparameterize(mu, log_var), mu, log_var

        # If not training, return the mean and standard deviation
        return mu, torch.exp(0.5 * log_var)

    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            mu, _ = self(x, training=False)
            dec_out = self.decoder(mu)
        return dec_out.detach()

    def _single_forward_step(self, batch, batch_index):

        x, = batch
        z, mu, log_var = self(x)
        x_hat = self.decoder(z)

        # Compute losses
        mse_loss = self.mse_loss(x_hat, x)  # reduction = mean already applies
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
            self._kl_ramp_strength = np.tanh(2.0 * epoch / self._kl_ramp_epochs)

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
        return DataLoader(ds, batch_size=2048, num_workers=12)

    def val_dataloader(self):
        print("val_dataloader called")
        ds = TensorDataset(self._X_val)
        return DataLoader(ds, batch_size=2048, num_workers=12)

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
