"""Variational autoencoder module class."""
from typing import Tuple

import torch.nn as nn
from torch import Tensor

from utils import init_weights
from vae.models.base import Decoder, Encoder, VAEConfig, VAEModel


class _Encoder(Encoder):
    def __init__(self, z_dim: int, nc: int) -> None:
        super().__init__()
        self.conv_stage = nn.Sequential(
            # input is (n_channels) x 28 x 28
            nn.Conv2d(nc, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 14 x 14
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 7 x 7
            nn.Conv2d(64 * 2, 64 * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 4 x 4
            nn.Flatten(),
        )
        # Encoder mean
        self.mean = nn.Linear(64 * 4 * 4 * 4, z_dim)
        # Encoder Variance log
        self.variance_log = nn.Linear(64 * 4 * 4 * 4, z_dim)

        # initialize weights
        self.conv_stage.apply(init_weights)
        self.mean.apply(init_weights)
        self.variance_log.apply(init_weights)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv_stage(x)
        return self.mean(x), self.variance_log(x)


class _Decoder(Decoder):
    def __init__(self, z_dim: int, nc: int) -> None:
        super().__init__()
        self.linear_stage = nn.Linear(z_dim, 64 * 4 * 4 * 4)
        self.conv_stage = nn.Sequential(
            # input is (64*4) x 4 x 4
            nn.ConvTranspose2d(64 * 4, 64 * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (64*2) x 7 x 7
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 14 x 14
            nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False),
            # state size. (n_channels) x 28 x 28
            nn.Sigmoid(),
        )

        # initialize weights
        self.linear_stage.apply(init_weights)
        self.conv_stage.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_stage(x)
        x = x.view(x.size(0), 64 * 4, 4, 4)
        return self.conv_stage(x)


class VAEModelV1(VAEModel):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__(config)
        self.encoder = _Encoder(config.z_dim, nc=1)
        self.decoder = _Decoder(config.z_dim, nc=1)


def _get_model_constructor() -> VAEModelV1:
    return VAEModelV1
