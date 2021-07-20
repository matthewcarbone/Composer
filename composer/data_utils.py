"""
Authors
-------
Cole Miles
    Cornell University, Department of Physics
Matthew R. Carbone
    Brookhaven National Laboratory, Computational Science Initiative
"""

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader


def numpy_to_FloatTensorLoader(
    X, batch_size=32, shuffle=True, pin_memory=True
):
    """Converts a numpy array to a torch FloatTensor, then to a
    TensorDataset, then to a DataLoader.

    Returns
    -------
    torch.utils.data.DataLoader
    """

    train = [torch.FloatTensor(X)]
    dataset = TensorDataset(*train)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=pin_memory
    )


class Splits:

    def __init__(self, data, seed=12345, splits=(0.8, 0.1, 0.1)):
        """
        Parameters
        ----------
        data : list
            A list of np.array objects. Could be [X] for an autoencoder, or
            [X, y] for a feedforward network, or [X, y, metadata]. Only
            requirement is that for [x, y, z], x.shape[0] == y.shape[0] == ...
        """

        if not isinstance(data, list):
            data = [data]

        # We only use a T/V/T split here
        assert len(splits) == 3
        assert sum(splits) == 1.0

        # Check data lengths are compatible
        self.data = data
        shapes = [xx.shape[0] for xx in self.data]
        assert all([xx == self.data[0].shape[0] for xx in shapes])

        # Make a new local RNG from the given seed
        rng = np.random.default_rng(seed)

        # Generate splits by slicing a random permutation
        length = self.data[0].shape[0]
        shuffled_indexes = rng.permutation(length)

        # Generate the splits themselves
        t_idx = int(np.ceil(splits[0] * length))
        v_idx = t_idx + int(np.ceil(splits[1] * length))
        train_idxs = shuffled_indexes[:t_idx]
        val_idxs = shuffled_indexes[t_idx:v_idx]
        test_idxs = shuffled_indexes[v_idx:]

        # Check all splits are disjoint
        assert set(train_idxs).isdisjoint(set(val_idxs))
        assert set(test_idxs).isdisjoint(set(val_idxs))
        assert set(train_idxs).isdisjoint(set(test_idxs))

        # Assign these lists as tensor objects
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs

    @property
    def train(self):
        return [xx[self.train_idxs, :] for xx in self.data]

    @property
    def valid(self):
        return [xx[self.val_idxs, :] for xx in self.data]

    @property
    def test(self):
        return [xx[self.test_idxs, :] for xx in self.data]

    @property
    def datasets(self):
        train = [torch.FloatTensor(xx) for xx in self.train]
        valid = [torch.FloatTensor(xx) for xx in self.valid]
        test = [torch.FloatTensor(xx) for xx in self.test]
        return {
            'train': TensorDataset(*train),
            'valid': TensorDataset(*valid),
            'test': TensorDataset(*test),
        }

    def get_loaders(self, batch_size=32, pin_memory=True):
        """Returns the train, valid and test loaders. Note the test set memory
        is never pinned."""

        ds = self.datasets
        return {
            'train': DataLoader(
                ds['train'], batch_size=batch_size, shuffle=True,
                pin_memory=pin_memory
            ),
            'valid': DataLoader(
                ds['valid'], batch_size=batch_size, shuffle=False,
                pin_memory=pin_memory
            ),
            'test': DataLoader(
                ds['test'], batch_size=batch_size, shuffle=False,
                pin_memory=False
            ),
        }


def random_in_interval(interval):
    return (interval[1] - interval[0]) * np.random.random() + interval[0]


def random_gaussians(
    dims=(500, 200), mu_range=(-1.0, 1.0), sd_range=(0.05, 0.1),
    height_range=(0.8, 1.0)
):
    """Returns a numpy array consisting of many Gaussians in which their
    maximum height has been scaled to 1."""

    grid = np.linspace(-1, 1, dims[1])

    return np.array([
        random_in_interval(height_range)
        * np.exp(
            -(random_in_interval(mu_range) - grid)**2 / 2.0
            / random_in_interval(sd_range)**2
        ) for _ in range(dims[0])
    ])
