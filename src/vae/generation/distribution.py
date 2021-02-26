"""Data generation by running forward pass through net."""
from typing import Tuple

import torch
from torch import Tensor
from sklearn.decomposition import PCA


def apply_distribution(real_images: Tensor, latents: Tensor, log_vars: Tensor, unique_latents: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
    mean, std = unique_latents.mean(dim=0, keepdim=True), unique_latents.std(dim=0, keepdim=True)
    new_latents = torch.cat([torch.normal(mean, std) for _ in range(latents.size(0))], dim=0)
    return new_latents, None, None
