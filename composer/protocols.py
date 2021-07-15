"""
Authors
-------
Cole Miles
    Cornell University, Department of Physics
Matthew R. Carbone
    Brookhaven National Laboratory, Computational Science Initiative
"""

import random
import time

import numpy as np
import torch
from torch.nn import MSELoss
# import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau

from composer.logger import Logger


# At the time that the module is called, this should be a global variable
CUDA_AVAIL = torch.cuda.is_available()


def vae_kl_loss(mu, logvar):
    """KL-Divergence of VAE ouputted (mu, logvar) with unit Gaussian."""

    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def seed_all(seed):
    """Helper function that seeds the random, numpy and torch modules all
    at once."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


class SerialTrainProtocol:
    """Base class for performing ML training. Attributes that are not also
    inputs to __init__ are listed below. A standard use case for this type of
    class might look like this:

    Example
    -------
    > x = TrainProtocol(train_loader, valid_loader)
    > x.initialize_model(model_params)
    > x.initialize_support()  # inits the optimizer, loss and scheduler
    > x.train()

    Attributes
    ----------
    trainLoader : torch.utils.data.dataloader.DataLoader
    validLoader : torch.utils.data.dataloader.DataLoader
    device : str
        Will be 'cuda:0' if at least one GPU is available, else 'cpu'.
    best_model_state_dict : Dict
        The torch model state dictionary corresponding to the best validation
        result. Used as a lightweight way to store the model parameters.
    """

    def __init__(
        self, root, trainLoader, validLoader, model, optimizer,
        criterion=MSELoss(), scheduler=None, seed=None
    ):
        """
        Parameters
        ----------
        trainLoader : torch.utils.data.dataloader.DataLoader
            The training loader object.
        validLoader : torch.utils.data.dataloader.DataLoader
            The cross validation (or testing) loader object.
        model : torch.nn.Module
            Class inheriting the torch.nn.Module back end.
        criterion : torch.nn._Loss
            Note, I think this is the correct object type. The criterion is the
            loss function.
        optimizer : torch.optim.Optimizer
            The numerical optimization protocol. Usually, we should choose
            Adam for this.
        scheduler : torch.optim.lr_scheduler
            Defines a protocol for training, usually for updating the learning
            rate. Default is None.
        seed : int
            Seeds random, numpy and torch. Default is None.
        """

        self.trainLoader = trainLoader
        self.validLoader = validLoader
        self.device = torch.device('cuda:0' if CUDA_AVAIL else 'cpu')
        seed_all(seed)
        self.model = model
        self.model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_model_state_dict = None
        self.log = Logger(root)

        # Log some information about the model and whatnot
        metadata = {
            "root": root,
            "train_device": str(self.device),
            "n_trainable": sum(
                param.numel() for param in self.model.parameters()
                if param.requires_grad
            ),
            "n_gpu": torch.cuda.device_count()
        }
        self.log.meta(metadata)

    def _eval_valid_pass(self, valLoader, cache=False):
        raise NotImplementedError

    def _train_single_epoch(self, clip):
        raise NotImplementedError

    def _eval_valid(self):
        """Similar to _train_single_epoch above, this method will evaluate a
        full pass on the validation data.

        Returns
        -------
        float
            The average loss on the validation data / batch.
        """

        self.model.eval()  # Freeze weights, set model to evaluation mode

        # Disable gradient updates.
        with torch.no_grad():
            losses, __ = self._eval_valid_pass(self.validLoader)

        return losses

    def _update_state_dict(self, best_valid_loss, valid_loss, epoch):
        """Updates the best_model_state_dict attribute if the valid loss is
        less than the best up-until-now valid loss.

        Parameters
        ----------
        best_valid_loss : float
            The best validation loss so far.
        valid_loss : float
            The current validation loss on the provided epoch.
        epoch : int
            The current epoch.

        Returns
        -------
        float
            min(best_valid_loss, valid_loss)
        """

        if valid_loss < best_valid_loss or epoch == 0:
            self.best_model_state_dict = self.model.state_dict()
            self.log.log(
                f"Val. Loss: {valid_loss:.05e} < Best Val. Loss "
                f"{best_valid_loss:.05e}"
            )
            self.log.log("Updating best_model_state_dict")
        else:
            self.log.log(f"Val. Loss: {valid_loss:.05e}")
            pass

        return min(best_valid_loss, valid_loss)

    def _step_scheduler(self, valid_loss):
        """Steps the scheduler and outputs information about the learning
        rate.

        Parameters
        ----------
        valid_loss : float
            The current epoch's validation loss.

        Returns
        -------
        float
            The current learning rate.
        """

        # Current learning rate
        clr = None
        if self.scheduler is not None:
            self.scheduler.step(np.sum(valid_loss))
            clr = self.scheduler.optimizer.param_groups[0]['lr']
            self.log.log(f"Learning rate {clr:.02e}")
        return clr

    def train(self, epochs, clip=None, print_every=10):
        """Executes model training.

        Parameters
        ----------
        epochs : int
            Number of full passes through the training data.
        clip : float, optional
            Gradient clipping.

        Returns
        -------
        train loss, validation loss, learning rates : list
        """

        # Keep track of the best validation loss so that we know when to save
        # the model state dictionary.
        best_valid_loss = float('inf')

        train_loss_list = []
        valid_loss_list = []
        learning_rates = []

        # Begin training
        self.epoch = 0
        while self.epoch < epochs:

            self.log.log(f"{self.epoch:03} begin")

            # Train a single epoch
            t0 = time.time()
            train_losses = self._train_single_epoch(clip)
            t_total = time.time() - t0

            # Evaluate on the validation data
            valid_losses = self._eval_valid()

            # Step the scheduler - returns the current learning rate (clr)
            total_valid_loss = np.sum(valid_losses).item()
            clr = self._step_scheduler(total_valid_loss)

            # Update the best state dictionary of the model for loading in
            # later on in the process
            best_valid_loss = self._update_state_dict(
                best_valid_loss, total_valid_loss, self.epoch
            )

            self.log.loss(self.epoch, t_total, train_losses, valid_losses, clr)

            train_loss_list.append(train_losses)
            valid_loss_list.append(valid_losses)
            learning_rates.append(clr)

            if (self.epoch + 1) % print_every == 0:
                e = self.epoch + 1
                print(
                    f"Done on epoch {e:03} in {t_total:.03f} s loss: "
                    f"{train_losses.sum():.02e} | {valid_losses.sum():.02e}"
                )

            self.epoch += 1

        self.model.load_state_dict(self.best_model_state_dict)

        return train_loss_list, valid_loss_list, learning_rates

    def eval(self, loader_override=None):
        """Systematically evaluates the validation, or dataset corresponding to
        the loader specified in the loader_override argument, dataset."""

        if loader_override is not None:
            # log.warning(
            #     "Default validation loader is overridden - ensure this is "
            #     "intentional, as this is likely evaluating on a testing set"
            # )
            pass

        loader = self.validLoader if loader_override is None \
            else loader_override

        # defaults.Result
        with torch.no_grad():
            losses, cache = self._eval_valid_pass(loader, cache=True)
        # log.info(f"Eval complete: loss {losses}")

        return losses, cache


class SerialVAETrainProtocol(SerialTrainProtocol):
    """Training protocol for VAE systems."""

    def __init__(
        self, root, trainLoader, validLoader, model, optimizer, criterion,
        scheduler=None, seed=None, kl_strength=0.1, ramp_kld=None
    ):
        """The kl_strength parameter is how much to weight the KL divergence,
        which enforces latent space activations to be part of a multivariate
        normal distribution."""

        super().__init__(
            root, trainLoader, validLoader, model, optimizer, criterion,
            scheduler, seed
        )
        self.kl_strength = kl_strength
        self.kl_ramp_epochs = None
        self.kl_prefactor = 1.0
        if ramp_kld is not None:
            self.kl_ramp_epochs = ramp_kld
            self.kl_prefactor = 0.0

    def _train_single_epoch(self, clip=None):
        """Executes the training of the model over a single full pass of
        training data.

        Parameters
        ----------
        clip : float
            Gradient clipping.

        Returns
        -------
        float
            The average training loss/batch.
        """

        self.model.train()  # Unfreeze weights, set model in train mode
        epoch_loss = []
        for batch, in self.trainLoader:

            self.optimizer.zero_grad()  # Zero the gradients

            batch = batch.to(device=self.device)  # Send batch to device

            # Run forward prop
            output, mu, log_var = self.model.forward(batch)
            r_loss = self.criterion(output, batch)
            kl_loss = self.kl_prefactor * self.kl_strength \
                * vae_kl_loss(mu, log_var)
            epoch_loss.append([
                r_loss.detach().item(), kl_loss.detach().item()
            ])
            loss = r_loss

            # Run back prop
            loss.backward()

            # Clip the gradients
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            # Step the optimizer
            self.optimizer.step()

        if self.kl_ramp_epochs is not None:
            self.kl_prefactor \
                = np.tanh(2.0 * self.epochs_ramped / self.kl_ramp_epochs)

        return np.mean(epoch_loss, axis=0)  # mean loss over this epoch

    def _eval_valid_pass(self, valLoader, cache=False):
        """Performs the for loop in evaluating the validation sets. This allows
        for interchanging the passed loaders as desired.

        Parameters
        ----------
        loader : torch.utils.data.dataloader.DataLoader
            Input loader to evaluate on.
        cache : bool
            If true, will save every output on the full pass to a dictionary so
            the user can evaluate every result individually.

        Returns
        -------
        float, List[defaults.Result]
            The average loss on the validation data / batch. Also returns a
            cache of the individual evaluation results if cache is True.
        """

        total_loss = []
        cache_list = []

        for batch, in valLoader:
            batch = batch.to(device=self.device)

            # Run forward prop
            output, mu, log_var = self.model.forward(batch)
            loss = self.criterion(output, batch)
            kl_loss = self.kl_prefactor * self.kl_strength \
                * vae_kl_loss(mu, log_var)
            total_loss.append((loss.detach().item(), kl_loss.detach().item()))
            loss += kl_loss

            if cache:
                output = output.cpu().detach().numpy()
                batch = batch.cpu().detach().numpy()
                mu = mu.cpu().detach().numpy()
                log_var = log_var.cpu().detach().numpy()
                cache_list_batch = [
                    (output[ii], batch[ii], mu[ii], log_var[ii])
                    for ii in range(len(batch))
                ]
                cache_list.extend(cache_list_batch)

        cache_list.sort(key=lambda x: np.abs(x[0] - x[1]).sum())
        return np.mean(total_loss, axis=0), cache_list
