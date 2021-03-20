"""Data generation by running forward pass through net."""
from typing import Tuple

import torch
from torch import Tensor


def add_noise(
    real_images: Tensor, latents: Tensor, log_vars: Tensor, std: float, alpha: float, **kwargs
) -> Tuple[Tensor, Tensor, Tensor]:
    eps = torch.empty_like(latents).normal_(0, std)
    return alpha * eps + latents, real_images, None


def normal_noise(
    real_images: Tensor, latents: Tensor, log_vars: Tensor, std: float, **kwargs
) -> Tuple[Tensor, Tensor, Tensor]:
    mean = torch.zeros_like(latents)
    std = torch.zeros_like(latents) + 1
    normal = torch.normal(mean, std)
    return normal, None, None

def uniform_noise(
    real_images: Tensor, latents: Tensor, log_vars: Tensor, **kwargs
) -> Tuple[Tensor, Tensor, Tensor]:
    z = torch.FloatTensor(size=latents.size(), device=latents.device).uniform_(-1.5, 1.5)
    return z, None, None
