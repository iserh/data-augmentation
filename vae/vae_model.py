"""Variational autoencoder module class."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    """Encoder module of VAE."""

    def __init__(self, z_dim: int) -> None:
        """Initialize encoder.

        Args:
            z_dim: Dimension of the latent space
        """
        super(Encoder, self).__init__()
        self.conv_stage = nn.Sequential(
            # Convolution 1 - (1, 28, 28) -> (32, 14, 14)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            # Convolution 2 - (32, 14, 14) -> (64, 7, 7)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            # Dense - (64, 7, 7) -> (64 * 7 * 7)
            nn.Flatten(),
        )
        # Encoder mean
        self.mean = nn.Linear(64 * 7 * 7, z_dim)
        # Encoder Variance log
        self.variance_log = nn.Linear(64 * 7 * 7, z_dim)

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

    def __init__(self, z_dim: int) -> None:
        """Initialize decoder.

        Args:
            z_dim: Dimension of the latent space
        """
        super(Decoder, self).__init__()
        self.linear_stage = nn.Linear(z_dim, 64 * 7 * 7)
        self.conv_stage = nn.Sequential(
            # Transpose convolution 1 - (64, 7, 7) -> (32, 14, 14)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Transpose convolution 2 - (32, 14, 14) -> (16, 28, 28)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # pixel-wise linear - (16, 28, 28) -> (1, 28, 28)
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Latent sample

        Returns:
            Reconstructed tensor
        """
        x = self.linear_stage(x)
        x = x.view(x.size(0), 64, 7, 7)
        return self.conv_stage(x)


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder module."""

    def __init__(self, z_dim: int) -> None:
        """Initialize variational autoencoder.

        Args:
            z_dim: Dimension of the latent space
        """
        super(VariationalAutoencoder, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Reconstructed tensor, mean, variance_log
        """
        mean, variance_log = self.encoder(x)
        if self.training:
            # reparameterization trick
            eps = torch.empty_like(variance_log).normal_()
            z = eps * (0.5 * variance_log).exp() + mean
        else:
            z = mean
        x_hat = self.decoder(z)

        return x_hat, mean, variance_log


def vae_loss(
    x_hat: Tensor,
    x_true: Tensor,
    mean: Tensor,
    variance_log: Tensor,
    beta: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Computes the vae loss.

    Args:
        x_hat: Reconstructed image
        x_true: True image
        mean: Means of latent space
        variance_log: Variance logs of latent space
        beta: Regulates KL-Divergence part

    Returns:
        BinaryCrossEntropy loss, KL-Divergence loss
    """
    bce = F.binary_cross_entropy(x_hat, x_true) * 28 * 28
    kld = beta * (variance_log.exp() + mean ** 2 - 1 - variance_log).sum(-1).mean()
    return bce, kld
