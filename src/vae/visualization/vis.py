"""Visualization functions."""
from typing import Optional

import mlflow
import numpy as np
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import Tensor


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
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
    # plot latents to figure
    if color_by_target:
        targets_unique = torch.unique(targets, sorted=True).int().tolist()
        for t in targets_unique:
            mask = targets.flatten() == t
            ax.scatter(*latents[mask].T, label=f"{t}")
    else:
        ax.scatter(*latents.T, label="z")
    plt.legend()
    if mlflow.active_run() is not None:
        mlflow.log_figure(fig, img_name + ".png")
    else:
        return fig
    plt.close()


def visualize_real_fake_images(
    reals: Tensor,
    fakes: Tensor,
    n: int,
    img_name: str = "real_fake",
    k: Optional[int] = None,
    indices: Optional[np.ndarray] = None,
    cols: int = 10,
) -> None:
    # k: number of generated fake images for each real image
    k = k or 1
    # if k > 1 duplicate reals corresponding to the amount of fakes
    dupl_reals = (
        reals.unsqueeze(1).expand((reals.size(0), k, *reals.size()[1:])).reshape(-1, *reals.size()[1:])
        if k > 1
        else reals
    )

    fig = plt.figure(figsize=(15, 15))
    fig.patch.set_alpha(0.0)
    n_plots = 3 if indices is not None else 2

    # Plot the real images
    plt.subplot(1, n_plots, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(dupl_reals[:n], padding=5, normalize=True, nrow=cols or k * 2),
            (1, 2, 0),
        )
    )

    # Plot the fake images
    plt.subplot(1, n_plots, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(fakes[:n], padding=5, normalize=True, nrow=cols or k * 2),
            (1, 2, 0),
        )
    )

    # plot the images that were used for generation
    if indices is not None:
        plt.subplot(1, n_plots, 3)
        plt.axis("off")
        plt.title("Images used for Generation")
        plt.imshow(
            np.transpose(
                vutils.make_grid(reals[indices[:n]], padding=5, normalize=True, nrow=cols or k * 2),
                (1, 2, 0),
            )
        )

    if mlflow.active_run() is not None:
        mlflow.log_figure(fig, img_name + ".png")
    else:
        return fig
    plt.close()


def visualize_images(images: Tensor, n: int, img_name: str = "images", cols: int = 10) -> None:
    fig = plt.figure(figsize=(15, 15))
    fig.patch.set_alpha(0.0)

    # Plot the real images
    plt.subplot(1, 1, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(images[:n], padding=5, normalize=True, nrow=cols),
            (1, 2, 0),
        )
    )

    if mlflow.active_run() is not None:
        mlflow.log_figure(fig, img_name + ".png")
    else:
        return fig
    plt.close()
