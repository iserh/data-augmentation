"""Variational autoencoder module class."""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        nn.init.dirac_(m.weight)
    elif type(m) == nn.ConvTranspose2d:
        nn.init.dirac_(m.weight)


class Encoder(nn.Module):
    """Encoder module of VAE."""

    def __init__(self, z_dim: int, n_channels: int) -> None:
        """Initialize encoder.

        Args:
            z_dim: Dimension of the latent space
        """
        super(Encoder, self).__init__()
        self.conv_stage = nn.Sequential(
            # input is (n_channels) x 64 x 64
            nn.Conv2d(n_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*8) x 4 x 4
            nn.Flatten(),
        )
        # Encoder mean
        self.mean = nn.Linear(64 * 8 * 4 * 4, z_dim)
        # Encoder Variance log
        self.variance_log = nn.Linear(64 * 8 * 4 * 4, z_dim)

        # initialize weights
        self.conv_stage.apply(init_weights)
        self.variance_log.apply(init_weights)
        self.mean.apply(init_weights)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Mean, variance_log
        """
        x = self.conv_stage(x)
        return self.mean(x), self.variance_log(x)


class Decoder(nn.Module):
    """Decoder module of VAE."""

    def __init__(self, z_dim: int, n_channels: int) -> None:
        """Initialize decoder.

        Args:
            z_dim: Dimension of the latent space
        """
        super(Decoder, self).__init__()
        self.linear_stage = nn.Linear(z_dim, 64 * 8 * 4 * 4)
        self.conv_stage = nn.Sequential(
            # input is (64*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (64*4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (64*2) x 16 x 16
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d(64, n_channels, 4, 2, 1, bias=False),
            # state size. (n_channels) x 64 x 64
            nn.Sigmoid(),
        )

        # initialize weights
        self.linear_stage.apply(init_weights)
        self.conv_stage.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Latent sample

        Returns:
            Reconstructed tensor
        """
        x = self.linear_stage(x)
        x = x.view(x.size(0), 64 * 8, 4, 4)
        return self.conv_stage(x)


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder module."""

    def __init__(self, z_dim: int, n_channels: int) -> None:
        """Initialize variational autoencoder.

        Args:
            z_dim: Dimension of the latent space
        """
        super(VariationalAutoencoder, self).__init__()
        self.code_paths = [__file__]
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim, n_channels)
        self.decoder = Decoder(z_dim, n_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Reconstructed tensor, mean, variance_log
        """
        # encode input tensor
        mean, variance_log = self.encoder(x)
        # reparameterization trick
        eps = torch.empty_like(variance_log).normal_()
        z = eps * (0.5 * variance_log).exp() + mean
        # decode latent tensor
        x_hat = self.decoder(z)
        # return everything
        return x_hat, mean, variance_log, z


class VAELoss:
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def __call__(
        self,
        x_true: Tensor,
        x_hat: Tensor,
        mean: Tensor,
        variance_log: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Computes the vae loss.

        Args:
            x_true: True image
            x_hat: Reconstructed image
            mean: Means of latent space
            variance_log: Variance logs of latent space

        Returns:
            BinaryCrossEntropy loss, KL-Divergence loss
        """
        bce = F.binary_cross_entropy(x_hat, x_true) * np.product(x_hat.size()[1:])
        kld = self.beta * (variance_log.exp() + mean ** 2 - 1 - variance_log).sum(-1).mean()
        return bce, kld
