"""Data generation by running forward pass through net."""
from typing import Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


def apply_interpolation(latents: Tensor, log_vars: Tensor, n_neighbors: int, alpha: float, **kwargs) -> Tuple[Tensor, Tensor]:
    # build nearest neighbour tree, k + 1 because the first neighbour is the point itself
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree").fit(latents)
    # get indices of k nearest neighbours for each latent vector
    _, indices = nbrs.kneighbors(latents)
    # the new latents and their root latents
    new_latents = torch.empty_like(latents)
    roots = torch.empty_like(latents)
    for i, idx in enumerate(indices):
        # choose one of the k nearest neighbors (ignore first one because its the latent vector itself)
        neighbor = np.random.choice(idx[1:])
        # interpolate
        new_latents[i] = (neighbor - latents[i]) * alpha + latents[i]
        # root is the neighbor
        roots[i] = neighbor

    return new_latents, roots


def apply_extrapolation(latents: Tensor, log_vars: Tensor, n_neighbors: int, alpha: float, **kwargs) -> Tuple[Tensor, Tensor]:
    return apply_interpolation(latents, log_vars, n_neighbors, -alpha, **kwargs)


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

    return torch.stack(interpolated, dim=0), torch.stack(corresponding_targets, dim=0)


def interpolate_along_dimension(z: Tensor, n_steps: int) -> Tensor:
    interpolated = []
    for dim in range(z.size(0)):
        other_dims = z.unsqueeze(0).expand((n_steps, z.size(0))).clone()
        x = np.linspace(z[dim] - 3, z[dim] + 3, n_steps)
        other_dims[:, dim] = torch.Tensor(x)
        interpolated.append(other_dims)
    return torch.stack(interpolated, dim=0)
