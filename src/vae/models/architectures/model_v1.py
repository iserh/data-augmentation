"""Variational autoencoder module class."""
from typing import Tuple

import torch.nn as nn
from torch import Tensor

from utils import init_weights
from vae.models.base import Decoder, Encoder, VAEConfig, VAEModel


class _Encoder(Encoder):
    def __init__(self, z_dim: int, nc: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # input size: (nc) x 28 x 28
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(nc, nc, 2, 1),
            nn.ReLU(inplace=True),
            # state size: (nc) x 28 x 28
            nn.Conv2d(nc, 64, 2, 2),
            nn.ReLU(inplace=True),
            # state size: (64) x 14 x 14
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (64) x 14 x 14
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (64) x 14 x 14
            nn.Flatten(),
            # state size: 64 * 14 * 14
            nn.Linear(64 * 14 * 14, 128)
            # output size: 128
        )
        # Encoder mean
        self.mean = nn.Linear(128, z_dim)
        # Encoder Variance log
        self.variance_log = nn.Linear(128, z_dim)

        # initialize weights
        self.encoder.apply(init_weights)
        self.mean.apply(init_weights)
        self.variance_log.apply(init_weights)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encoder(x)
        return self.mean(x), self.variance_log(x)


class _Decoder(Decoder):
    def __init__(self, z_dim: int, nc: int) -> None:
        super().__init__()
        self.upsampling = nn.Sequential(
            # input size: z_dim
            nn.Linear(z_dim, 128),
            # state size: 128
            nn.Linear(128, 64 * 14 * 14),
            # output size: 64 * 14 * 14
        )
        self.decoder = nn.Sequential(
            # input size: (64) x 14 x 14
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (64) x 14 x 14
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (64) x 14 x 14
            nn.ConvTranspose2d(64, 64, 3, 2),
            nn.ReLU(inplace=True),
            # state size: (64) x 29 x 29
            nn.ZeroPad2d((0, -2, 0, -2)),
            # state size: (64) x 27 x 27 ! TODO: better solution for padding
            nn.ConvTranspose2d(64, nc, 2, 1),
            nn.Sigmoid(),
            # output size: (nc) x 28 x 28
        )

        # initialize weights
        self.upsampling.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsampling(x)
        x = x.view(x.size(0), 64, 14, 14)
        return self.decoder(x)


class VAEModelV1(VAEModel):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__(config)
        self.encoder = _Encoder(config.z_dim, nc=1)
        self.decoder = _Decoder(config.z_dim, nc=1)


def _get_model_constructor() -> VAEModelV1:
    return VAEModelV1
