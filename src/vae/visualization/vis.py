"""Visualization functions."""
from typing import Optional

import numpy as np
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import Tensor

from utils import mlflow_active, mlflow_available

if mlflow_available():
    import mlflow


def visualize_latents(
    latents: Tensor,
    pca: Optional[PCA] = None,
    labels: Optional[Tensor] = None,
    color_by_label: bool = False,
    **kwargs,
) -> None:
    if pca is not None:
        latents = pca.transform(latents)
    # create pyplot figure and axes
    fig = plt.figure(figsize=(16, 16))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
    # plot latents to figure
    if color_by_label:
        targets_unique = torch.unique(labels, sorted=True).int().tolist()
        for t in targets_unique:
            mask = labels.flatten() == t
            ax.scatter(*latents[mask].T, label=f"{t}")
    else:
        ax.scatter(*latents.T, label="z")
    plt.legend()
    if mlflow_active():
        mlflow.log_figure(fig, kwargs.get("filename", "latents.png"))
    else:
        plt.show(fig)
        plt.close(fig)


def visualize_images(
    images: Tensor,
    n: int,
    heritages: Optional[Tensor] = None,
    partners: Optional[Tensor] = None,
    cols: int = 10,
    **kwargs,
) -> None:
    fig = plt.figure(figsize=(15, 15))
    fig.patch.set_alpha(0.0)
    if heritages is None or partners is None:
        n_plots = 1 if heritages is None and partners is None else 2
    else:
        n_plots = 3

    # Plot the images
    plt.subplot(1, n_plots, 1)
    plt.axis("off")
    plt.title(kwargs.get("img_title", "Images"))
    plt.imshow(
        np.transpose(
            vutils.make_grid(images[:n], padding=5, normalize=True, nrow=cols),
            (1, 2, 0),
        )
    )

    # Plot the heritages
    if heritages is not None:
        plt.subplot(1, n_plots, 2)
        plt.axis("off")
        plt.title(kwargs.get("heritage_title", "Heritages"))
        plt.imshow(
            np.transpose(
                vutils.make_grid(heritages[:n], padding=5, normalize=True, nrow=cols),
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

    if mlflow_active():
        mlflow.log_figure(fig, kwargs.get("filename", "images.png"))
    else:
        plt.show(fig)
        plt.close(fig)
