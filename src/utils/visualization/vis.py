"""Visualization functions."""
from typing import Optional

import numpy as np
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import Tensor
from torchvision import transforms

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
            xlim (Tuple[float, float]): Limits of the plot (x)
            ylim (Tuple[float, float]): Limits of the plot (y)
            title
    """
    if pca is not None:
        points = pca.transform(points)
    # create pyplot figure and axes
    fig = plt.figure(figsize=(6.4, 6.4), tight_layout=True)
    plt.xlim(kwargs.get("xlim", (-4, 4)))
    plt.ylim(kwargs.get("ylim", (-4, 4)))
    if kwargs.get("title", False):
        plt.title(kwargs["title"], fontdict={"fontsize": 32})
    # plot latents to figure
    if labels is not None:
        classes = torch.unique(labels, sorted=True).int().tolist()
        for c in classes:
            mask = labels.flatten() == c
            plt.scatter(*points[mask].T, label=f"{c}", s=5)
        plt.legend()
    else:
        plt.scatter(*points.T, s=5)
    if mlflow_active():
        mlflow.log_figure(fig, kwargs.get("filename", "latents.pdf"))
    else:
        if kwargs.get("filename", False):
            plt.savefig(kwargs["filename"])
        else:
            plt.show()
    plt.close()


def plot_images(
    images: Tensor,
    n: int,
    origins: Optional[Tensor] = None,
    others: Optional[Tensor] = None,
    cols: int = 5,
    grayscale: bool = False,
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
    # temporary
    if images.size(1) == 1:
        grayscale = True
        kwargs["cmap"] = kwargs.get("cmap", "gray_r")

    n_plots = 1
    if origins is not None:
        n_plots += 1
    if others is not None:
        n_plots += 1

    if n_plots > 1:
        fig, axes = plt.subplots(ncols=n_plots, tight_layout=True)
    else:
        fig = plt.figure(tight_layout=True)
        axes = [plt.axes()]

    to_grayscale = transforms.Grayscale(1) if grayscale else lambda x: x
    i = 0

    # Plot the images
    axes[i].set_title(kwargs.get("images_title", "Images"))
    axes[i].axis("off")
    axes[i].imshow(
        np.transpose(
            to_grayscale(vutils.make_grid(images[:n], padding=5, normalize=True, nrow=cols)),
            (1, 2, 0),
        ),
        interpolation="nearest",
        cmap=kwargs.get("cmap", None),
    )

    # Plot the heritages
    if origins is not None:
        i += 1
        axes[i].set_title(kwargs.get("origins_title", "Origins"))
        axes[i].axis("off")
        axes[i].imshow(
            np.transpose(
                to_grayscale(vutils.make_grid(origins[:n], padding=5, normalize=True, nrow=cols)),
                (1, 2, 0),
            ),
            interpolation="nearest",
            cmap=kwargs.get("cmap", None),
        )

    # plot the images that were used for generation
    if others is not None:
        i += 1
        axes[i].set_title(kwargs.get("others_title", "Others"))
        axes[i].axis("off")
        axes[i].imshow(
            np.transpose(
                to_grayscale(vutils.make_grid(others[:n], padding=5, normalize=True, nrow=cols)),
                (1, 2, 0),
            ),
            interpolation="nearest",
            cmap=kwargs.get("cmap", None),
        )

    if mlflow_active():
        mlflow.log_figure(fig, kwargs.get("filename", "images.pdf"))
    else:
        if kwargs.get("filename", False):
            plt.savefig(kwargs["filename"])
        else:
            plt.show()
    plt.close()
