"""Data generation by running forward pass through net."""
from typing import Optional, Tuple

import torch
from torch import Tensor


class Noise:
    def __init__(
        self, alpha: float, k: int, std: float, return_indices: bool = False, indices_before: Optional[Tensor] = None
    ) -> None:
        self.alpha = alpha
        self.k = k
        self.std = std
        self.return_indices = return_indices
        self.indices_before = indices_before

    def __call__(self, z: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z_ = z.unsqueeze(1).expand(z.size(0), self.k, *z.size()[1:])
        y_ = y.unsqueeze(1).expand(y.size(0), self.k, *y.size()[1:])
        indices_before = (
            torch.tensor(self.indices_before) if self.indices_before is not None else torch.arange(0, z.size(0), 1)
        )
        new_indices = indices_before.unsqueeze(1).expand(indices_before.size(0), self.k)
        normal = torch.empty_like(z_).normal_(0, self.std)
        z_ = z_ + self.alpha * normal

        if self.return_indices:
            return z_.reshape(-1, *z.size()[1:]), y_.reshape(-1, *y.size()[1:]), new_indices.reshape(-1)
        else:
            return z_.reshape(-1, *z.size()[1:]), y_.reshape(-1, *y.size()[1:])
