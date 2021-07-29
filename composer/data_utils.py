import torch


def gaussian(x, m, s):
    """Returns a Gaussian on a grid, x, with mean m and standard deviation
    s. The Gaussian will have max height of 1 when x = m."""

    return torch.exp(-(x - m)**2 / (2.0 * s**2))


def random_gaussian_dataset(n_samples=2**11 * 5, l_signal=128):
    x = torch.linspace(-2.5, 2.5, l_signal).expand(n_samples, l_signal)

    mu = torch.rand(size=(n_samples, 1)) * 1.6 - 0.8
    sig = (torch.rand(size=(n_samples, 1)) * 0.99 + 0.01) * 0.75

    noise = (torch.rand(size=(n_samples, l_signal)) - 0.5) * 1e-2
    data = gaussian(x, mu, sig) + noise

    # Normalize to (0, 1)
    return torch.FloatTensor(data), mu, sig


GRID = torch.linspace(-1, 1, 128)
