"""Data generation by running forward pass through net."""
from typing import Tuple

import torch
from torch import Tensor


def add_noise(latents: Tensor, log_vars: Tensor, std: float, alpha: float, **kwargs) -> Tuple[Tensor, Tensor]:
    eps = torch.empty_like(latents).normal_(0, std)
    return alpha * eps + latents, latents


def normal_noise(latents: Tensor, log_vars: Tensor, std: float, alpha: float, **kwargs) -> Tuple[Tensor, Tensor]:
    mean = torch.zeros_like(latents)
    std = torch.zeros_like(latents) + 1
    normal = torch.normal(mean, std)
    return normal, normal
