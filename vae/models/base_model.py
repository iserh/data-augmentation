"""Variational autoencoder module base classes."""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from utils.integrations import ExperimentName, load_pytorch_model
from utils.models import PretrainedConfig


@dataclass
class VAEConfig(PretrainedConfig):
    z_dim: int
    beta: float


class EncoderBaseModel(nn.Module):
    def __init__(self, z_dim: int, nc: int) -> None:
        super(EncoderBaseModel, self).__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()


class DecoderBaseModel(nn.Module):
    def __init__(self, z_dim: int, nc: int) -> None:
        super(DecoderBaseModel, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class VAEBaseModel(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super(VAEBaseModel, self).__init__()
        self.code_paths = [__file__]
        self.z_dim = z_dim
        self.encoder: nn.Module = NotImplemented
        self.decoder: nn.Module = NotImplemented

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # encode input tensor
        mean, log_var = self.encoder(x)
        # reparameterization trick
        eps = torch.empty_like(log_var).normal_()
        z = eps * (0.5 * log_var).exp() + mean
        # decode latent tensor
        x_hat = self.decoder(z)
        # return everything
        return x_hat, mean, log_var

    @staticmethod
    def from_pretrained(config: VAEConfig) -> "VAEBaseModel":
        return load_pretrained_model(config)


def load_pretrained_model(config: VAEConfig) -> VAEBaseModel:
    return load_pytorch_model(ExperimentName.VAETrain, config)
