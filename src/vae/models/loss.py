"""Variational autoencoder loss criterion."""
from typing import Any, Tuple

import numpy as np
import torch.nn.functional as F
from torch import Tensor


class VAELossOutput:
    def __init__(self, r_loss: Tensor, kl_loss: Tensor, beta_norm: float) -> None:
        self.r_loss = r_loss
        self.kl_loss = kl_loss
        self.beta_norm = beta_norm

    def backward(self) -> None:
        (self.r_loss + self.beta_norm * self.kl_loss).backward()

    def item(self) -> Tuple[float, float]:
        return VAELossOutput(self.r_loss.item(), self.kl_loss.item(), self.beta_norm)

    def __add__(self, other: "VAELossOutput") -> "VAELossOutput":
        if not isinstance(other, VAELossOutput):
            raise TypeError(f"unsupported operand type(s) for +: ‘{type(other)}’ and ‘VAELossOutput’")
        self.r_loss += other.r_loss
        self.kl_loss += other.kl_loss
        return self

    def __radd__(self, other: Any) -> "VAELossOutput":
        if other == 0:
            return self
        else:
            return self.__add__(other)


class VAELoss:
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def __call__(self, reconstruction: Tensor, target: Tensor, m: Tensor, log_v: Tensor) -> VAELossOutput:
        input_dim = np.product(reconstruction.size()[1:])
        r_l = F.binary_cross_entropy(reconstruction, target) * input_dim
        kld = ((log_v.exp() + m ** 2 - 1 - log_v).sum(-1) * 0.5).mean()
        return VAELossOutput(r_l, kld, beta_norm=self.beta)
