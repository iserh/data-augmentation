"""Variational autoencoder loss criterion."""
from typing import Tuple

import numpy as np
import torch.nn as nn
from torch import Tensor


class VAELoss:
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta
        self.reconstruction_loss = nn.BCELoss()
        self.kl_divergence = lambda m, log_v: (log_v.exp() + m ** 2 - 1 - log_v).sum(-1).mean()

    def __call__(self, reconstruction: Tensor, target: Tensor, m: Tensor, log_v: Tensor) -> Tuple[Tensor, Tensor]:
        r_l = self.reconstruction_loss(reconstruction, target) * np.product(reconstruction.size()[1:])
        kld = self.beta * self.kl_divergence(m, log_v)
        return r_l, kld
