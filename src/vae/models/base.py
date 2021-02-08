"""Variational autoencoder module base classes."""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from .loss import VAELoss, VAELossOutput

import torch
import torch.nn as nn
from torch import Tensor

from utils.integrations import ExperimentName, load_pytorch_model
from utils.models import PretrainedConfig


@dataclass
class VAEConfig(PretrainedConfig):
    z_dim: int = 2
    beta: float = 1.0
    compute_loss: bool = True


@dataclass
class VAEOutput:
    reconstructed: Tensor
    mean: Tensor
    log_var: Tensor
    z: Tensor
    loss: Optional[VAELossOutput] = None


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: Tensor) -> Tensor:
        raise NotImplementedError()


class VAEModel(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super(VAEModel, self).__init__()
        self.code_paths = [__file__]
        self.config = config
        self.criterion = VAELoss(beta=config.beta)
        self.encoder: nn.Module = NotImplemented
        self.decoder: nn.Module = NotImplemented

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # encode input tensor
        mean, log_var = self.encoder(x)
        # reparameterization trick
        eps = torch.empty_like(log_var).normal_()
        z = eps * (0.5 * log_var).exp() + mean
        # decode latent tensor
        x_hat = self.decoder(z)
        # compute loss
        loss = self.criterion(x_hat, x, mean, log_var) if self.config.compute_loss else None
        return VAEOutput(
            reconstructed=x_hat,
            mean=mean,
            log_var=log_var,
            z=z,
            loss=loss,
        )

    @staticmethod
    def from_pretrained(config: VAEConfig) -> "VAEModel":
        return load_pretrained_model(config)


def load_pretrained_model(config: VAEConfig) -> VAEModel:
    return load_pytorch_model(ExperimentName.VAETrain, config)
