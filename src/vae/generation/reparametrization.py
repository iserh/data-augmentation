"""Data generation by running forward pass through net."""
from typing import Tuple

import torch
from torch import Tensor


def apply_reparametrization(latents: Tensor, log_vars: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
    eps = torch.empty_like(log_vars).normal_(0, 1)
    return eps * (0.5 * log_vars).exp() + latents, latents
