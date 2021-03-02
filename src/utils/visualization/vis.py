"""Visualization functions."""
from typing import Optional

import numpy as np
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import Tensor

from utils.mlflow import mlflow_active, mlflow_available

if mlflow_available():
    import mlflow


def plot_points(
    points: Tensor,
    pca: Optional[PCA] = None,
    labels: Optional[Tensor] = None,
    **kwargs,
) -> None:
    """Scatter plot of points. If dimension is higher than 2 the pca argument is required.

    Plots to a file in active artifact store (if mlflow is running), otherwise just shows the figure.

    Args:
        points (Tensor): Points to plot
        pca (Optional[PCA]): PCA to reduce dimension
        labels (Optional[Tensor]): Labels for coloring/legend
        **kwargs: Keyword arguments
            filename (str): Name of the output file
    """
    if pca is not None:
        latents = pca.transform(points)
    # create pyplot figure and axes
    fig = plt.figure(figsize=(16, 16))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
    # plot latents to figure
    if labels is not None:
        classes = torch.unique(labels, sorted=True).int().tolist()
        for c in classes:
            mask = labels.flatten() == c
            ax.scatter(*latents[mask].T, label=f"{c}")
        plt.legend()
    else:
        ax.scatter(*latents.T)
    if mlflow_active():
        mlflow.log_figure(fig, kwargs.get("filename", "latents.png"))
    else:
        plt.show()
    plt.close()


def plot_images(
    images: Tensor,
    n: int,
    origins: Optional[Tensor] = None,
    others: Optional[Tensor] = None,
    cols: int = 5,
    **kwargs,
) -> None:
    """Plot images in a grid.

    Plots to a file in active artifact store (if mlflow is running), otherwise just shows the figure.

    Args:
        images (Tensor): Images to plot
        n (int): Limit amount of images displayed
        origins (Optional[Tensor]): Side by side view of another image tensor
        others (Optional[Tensor]): Side by side view of another image tensor
        cols (int): Number of columns in grid
        **kwargs: Keyword arguments
            filename (str): Name of the output file
            images_title (str): Title of the images subplot
            origins_title (str): Title of the origins subplot
            others_title (str): Title of the others subplot
    """
    fig = plt.figure(figsize=(15, 15))
    fig.patch.set_alpha(0.0)
    if origins is None or others is None:
        n_plots = 1 if origins is None and others is None else 2
    else:
        n_plots = 3

    # Plot the images
    plt.subplot(1, n_plots, 1)
    plt.axis("off")
    plt.title(kwargs.get("images_title", "Images"))
    plt.imshow(
        np.transpose(
            vutils.make_grid(images[:n], padding=5, normalize=True, nrow=cols),
            (1, 2, 0),
        )
    )

    # Plot the heritages
    if origins is not None:
        plt.subplot(1, n_plots, 2)
        plt.axis("off")
        plt.title(kwargs.get("origins_title", "Origins"))
        plt.imshow(
            np.transpose(
                vutils.make_grid(origins[:n], padding=5, normalize=True, nrow=cols),
                (1, 2, 0),
            )
        )

    # plot the images that were used for generation
    if others is not None:
        plt.subplot(1, n_plots, 3)
        plt.axis("off")
        plt.title(kwargs.get("others_title", "Others"))
        plt.imshow(
            np.transpose(
                vutils.make_grid(others[:n], padding=5, normalize=True, nrow=cols),
                (1, 2, 0),
            )
        )

    if mlflow_active():
        mlflow.log_figure(fig, kwargs.get("filename", "images.png"))
    else:
        plt.show()
    plt.close()
