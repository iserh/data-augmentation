"""Visualization functions."""
from typing import Optional
from utils.mlflow_utils import ExperimentTypes

import mlflow
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import Tensor
import numpy as np
import torchvision.utils as vutils


def visualize_latents(
    latents: Tensor,
    pca: Optional[PCA] = None,
    targets: Optional[Tensor] = None,
    color_by_target: bool = False,
    img_name: str = "latents",
) -> None:
    if pca is not None:
        latents = pca.transform(latents)
    # create pyplot figure and axes
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
    # plot latents to figure
    if color_by_target:
        targets_unique = torch.unique(targets, sorted=True).int().tolist()
        for t in targets_unique:
            mask = targets[:, 0] == t
            ax.scatter(*latents[mask].T, label=f"{t}")
    else:
        ax.scatter(*latents.T, label="z")
    plt.legend()
    mlflow.log_figure(fig, img_name + ".png")
    plt.close()


def visualize_real_fake_images(reals: Tensor, fakes: Tensor, n: int, img_name: str = "real_fake") -> None:
    fig = plt.figure(figsize=(15, 15))

    # Plot the real images
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(reals[:n], padding=5, normalize=True, nrow=6), (1, 2, 0),))

    # Plot the fake images
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(fakes[:n], padding=5, normalize=True, nrow=6), (1, 2, 0),))

    mlflow.log_figure(fig, img_name + ".png")
    plt.close()


if __name__ == "__main__":
    from utils.mlflow_utils import Roots, Experiment

    mlflow.set_tracking_uri(Roots.TEST.value)
    exp = Experiment(ExperimentTypes.FeatureSpace)

    with exp.new_run("test") as run:
        latents = torch.empty((200, 2)).normal_(1, 1)
        pca = PCA(2).fit(latents) if latents.size(1) > 0 else None
        visualize_latents(latents, pca)
