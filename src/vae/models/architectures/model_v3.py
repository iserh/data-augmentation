"""Variational autoencoder module class."""
from typing import Tuple

import torch.nn as nn
from torch import Tensor

from utils import init_weights
from vae.models.base import Decoder, Encoder, VAEConfig, VAEModel


class _Encoder(Encoder):
    def __init__(self, z_dim: int, num_features: int) -> None:
        super(_Encoder, self).__init__()
        self.linear_stage = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        # Encoder mean
        self.mean = nn.Linear(64, z_dim)
        # Encoder Variance log
        self.variance_log = nn.Linear(64, z_dim)

        # initialize weights
        self.linear_stage.apply(init_weights)
        self.mean.apply(init_weights)
        self.variance_log.apply(init_weights)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.linear_stage(x)
        return self.mean(x), self.variance_log(x)


class _Decoder(Decoder):
    def __init__(self, z_dim: int, num_features: int) -> None:
        super(_Decoder, self).__init__()
        self.linear_stage = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, num_features),
            nn.Sigmoid(),
        )

        # initialize weights
        self.linear_stage.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_stage(x)


class VAEModelV3(VAEModel):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__(config)
        self.encoder = _Encoder(config.z_dim, num_features=8)
        self.decoder = _Decoder(config.z_dim, num_features=8)


def _get_model_constructor() -> VAEModelV3:
    return VAEModelV3
