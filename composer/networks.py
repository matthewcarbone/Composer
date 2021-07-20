"""
Authors
-------
Cole Miles
    Cornell University, Department of Physics
Matthew R. Carbone
    Brookhaven National Laboratory, Computational Science Initiative
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """Feedforward neural network model with a final linear activation."""

    def __init__(self, n_in, hidden_sizes, n_out, dropout, activation):
        """
        Parameters
        ----------
        n_in : int
            Number of input neurons.
        hidden_sizes : list
            A list of integers in which each entry represents a hidden layer
            of size hidden_sizes[ii].
        n_out : int
            Number of output neurons.
        dropout : float
            Dropout applied to all layers.
        activation : {'relu', 'leaky_relu'}
            Activation function.
        """

        super().__init__()

        if not isinstance(hidden_sizes, list):
            critical = "Parameter hidden_sizes must be of type List[int]"
            raise ValueError(critical)

        self.input_layer = torch.nn.Linear(n_in, hidden_sizes[0])

        hidden_layers = []
        hidden_bn_layers = []

        for ii in range(0, len(hidden_sizes) - 1):
            hidden_layers.append(torch.nn.Linear(
                hidden_sizes[ii], hidden_sizes[ii + 1], bias=False
            ))

        for ii in range(0, len(hidden_sizes) - 1):
            hidden_bn_layers.append(
                torch.nn.BatchNorm1d(hidden_sizes[ii + 1])
            )

        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.hidden_bn_layers = nn.ModuleList(hidden_bn_layers)

        self.output_layer = torch.nn.Linear(
            hidden_sizes[-1], n_out
        )

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError("Unknown activation specified")

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Forward propagation for the Encoder. Applies activations to every
        layer, in addition to dropout (except to the output layer)."""

        x = self.dropout(self.activation(self.input_layer(x)))
        for layer, bn_layer in zip(self.hidden_layers, self.hidden_bn_layers):
            x = self.dropout(self.activation(bn_layer(layer(x))))
        return self.output_layer(x)


class Encoder(NeuralNetwork):
    """Encodes an input vector into a [lower dimensional] latent space."""

    def __init__(
        self, input_size, hidden_sizes, latent_space_size, dropout=0.0,
        activation='leaky_relu'
    ):
        super().__init__(
            input_size, hidden_sizes, latent_space_size, dropout, activation
        )


class Decoder(NeuralNetwork):
    """Reconstruct an input vector from the latent space."""

    def __init__(
        self, latent_space_size, hidden_sizes, output_size, dropout=0.0,
        activation=torch.relu
    ):
        super().__init__(
            latent_space_size, hidden_sizes, output_size, dropout, activation
        )


def reparameterize(mu, log_var):
    """Implements the 'reparameterization trick'

    Parameters
    ----------
    mu : torch.Tensor
        Mean from the encoder's latent space.
    log_var : torch.Tensor
        Log variance from the encoder's latent space.

    Returns
    -------
    torch.Tensor
    """

    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    sample = mu + eps * std
    return sample


class VariationalAutoencoder(nn.Module):
    """Vector-to-vector (fixed-length) variational autoencoder. The autoencoder
    is symmetric. Note that the number of outputs from the encoder is actually
    2 x latent space size to account for the mean and standard deviation of
    the sampling."""

    def __init__(
        self, input_size, hidden_sizes, latent_space_size,
        dropout=0.0, activation="relu", final_activation="softplus"
    ):
        """Initializer.
        Parameters
        ----------
        input_size : int
            The size of the input and output of the autoencoder.
        hidden_sizes : list
            A list of integers determining the number of and number of neurons
            per layer.
        latent_space_size : int
        """

        # Used for model save/reload
        self.model_args = {
            'name': str(self.__class__),
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'latent_space_size': latent_space_size,
            'dropout': dropout,
            'activation': activation,
            'final_activation': final_activation
        }

        super().__init__()

        self.encoder = Encoder(
            input_size, hidden_sizes, 2 * latent_space_size, dropout=dropout,
            activation=activation
        )
        self.decoder = Decoder(
            latent_space_size, list(reversed(hidden_sizes)), input_size,
            dropout=dropout, activation=activation
        )
        self.latent_space_size = latent_space_size

        self.final_activation = None
        if final_activation == "softplus":
            self.final_activation = nn.Softplus()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()

    def forward(self, x):
        """Forward propagation for the autoencoder."""

        x = self.encoder(x)
        x = x.view(-1, 2, self.latent_space_size)
        # Use first set of outputs as mean of distributions
        mu = x[:, 0, :]
        # And the second set as the log variances
        log_var = x[:, 1, :]
        z = reparameterize(mu, log_var)
        out = self.decoder(z)
        if self.final_activation is not None:
            out = self.final_activation(out)
        return out, mu, log_var


def load_VariationalAutoencoder(path, model_kwargs):
    """Loads the VAE model from the checkpoint provided."""

    device = torch.device('cpu')
    if "name" in list(model_kwargs.keys()):
        model_kwargs.pop("name")
    model = VariationalAutoencoder(**model_kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
