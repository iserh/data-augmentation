"""Data generation by running forward pass through net."""
from typing import Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


class Interpolation:
    def __init__(self, alpha: float, k: int = 3, return_indices: bool = False) -> None:
        self.alpha = alpha
        self.k = k
        self.return_indices = return_indices

    def __call__(self, z: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # build nearest neighbour tree, k + 1 because the first neighbour is the point itself
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm="ball_tree").fit(z)
        # get indices of k nearest neighbours for each latent vector
        _, indices = nbrs.kneighbors(z)
        # generate k new latents for each original latent vector
        # by interpolating between the k'th nearest neighbour
        z_ = torch.empty((z.size(0), self.k, *z.size()[1:]), device=z.device, dtype=z.dtype)
        y_ = torch.empty((y.size(0), self.k, *y.size()[1:]), device=y.device, dtype=y.dtype)
        for i in range(z.size(0)):
            # each latent vector generates 'n_neighbor' new latent vectors
            for j, k in enumerate(indices[i][1:]):
                # interpolate between latent vector and the k'th nearest neighbour
                z_[i, j] = (z[k] - z[i]) * self.alpha + z[i]
                # save the target too
                y_[i, j] = y[i]
        # reshape
        z_ = z_.reshape(-1, z.size(-1))
        y_ = y_.flatten()
        # return new modified latents and the corresponding targets as tensors
        return (z_, y_, indices.reshape(-1)) if self.return_indices else (z_, y_)


def interpolate_along_class(latents: Tensor, targets: Tensor, n_steps: int) -> Tensor:
    unique_targets = torch.unique(targets, sorted=True).tolist()
    interpolated, corresponding_targets = [], []
    for i in unique_targets:
        mask = targets == i
        pca = PCA(1).fit(latents[mask.flatten()])
        pc = pca.transform(latents)
        x = np.expand_dims(np.linspace(pc.min(), pc.max(), n_steps), axis=1)
        interpolated.append(torch.Tensor(pca.inverse_transform(x)))
        corresponding_targets.append(torch.Tensor([i] * n_steps))

    return torch.cat(interpolated, dim=0), torch.cat(corresponding_targets, dim=0)


def interpolate_along_dimension(z: Tensor, n_steps: int) -> Tensor:
    interpolated = []
    for dim in range(z.size(0)):
        other_dims = z.unsqueeze(0).expand((n_steps, z.size(0))).clone()
        x = np.linspace(z[dim] - 3, z[dim] + 3, n_steps)
        other_dims[:, dim] = torch.Tensor(x)
        interpolated.append(other_dims)
    return torch.cat(interpolated, dim=0)
