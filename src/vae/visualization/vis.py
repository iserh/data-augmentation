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


def visualize_heritages_partners(
    images: Tensor,
    heritages: Tensor,
    partners: Tensor,
    n: int,
    cols: int = 10,
    **kwargs,
) -> None:
    fig = plt.figure(figsize=(15, 15))
    fig.patch.set_alpha(0.0)
    n_plots = 3 if partners is not None else 2

    # Plot the real images
    plt.subplot(1, n_plots, 1)
    plt.axis("off")
    plt.title(kwargs.get("heritage_title", "Heritages"))
    plt.imshow(
        np.transpose(
            vutils.make_grid(heritages[:n], padding=5, normalize=True, nrow=cols),
            (1, 2, 0),
        )
    )

    # Plot the fake images
    plt.subplot(1, n_plots, 2)
    plt.axis("off")
    plt.title(kwargs.get("img_title", "Images"))
    plt.imshow(
        np.transpose(
            vutils.make_grid(images[:n], padding=5, normalize=True, nrow=cols),
            (1, 2, 0),
        )
    )

    # plot the images that were used for generation
    if partners is not None:
        plt.subplot(1, n_plots, 3)
        plt.axis("off")
        plt.title(kwargs.get("partner_title", "Partners"))
        plt.imshow(
            np.transpose(
                vutils.make_grid(partners[:n], padding=5, normalize=True, nrow=cols),
                (1, 2, 0),
            )
        )

    if mlflow.active_run() is not None:
        mlflow.log_figure(fig, kwargs.get("filename", "heritages_partners") + ".png")
    else:
        return fig
    plt.close()


def visualize_images(images: Tensor, n: int, cols: int = 10, img_name: str = "images", title: str = "Images") -> None:
    fig = plt.figure(figsize=(15, 15))
    fig.patch.set_alpha(0.0)

    # Plot the real images
    plt.subplot(1, 1, 1)
    plt.axis("off")
    plt.title(title)
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
