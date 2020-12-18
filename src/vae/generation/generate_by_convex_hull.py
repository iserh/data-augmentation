"""Data generation."""
import pickle
from pathlib import Path
from typing import List, Tuple

from utils.plotting import reshape_to_img
from vae.model_setup import alpha, beta, epochs, model_setup, z_dim
from vae.vae_model_v1 import VariationalAutoencoder

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import get_cmap
from numpy.linalg import det
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import dirichlet
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data.dataloader import DataLoader


def dist_in_hull(points: np.ndarray, n: int) -> np.ndarray:
    """Sample uniformly from convex hull of points.

    Args:
        points (np.ndarray): Points to sample from ch
        n (int): Number of samples

    Returns:
        np.ndarray: Samples
    """
    dims = points.shape[-1]
    hull = points[ConvexHull(points).vertices]
    deln = points[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)
    sample = np.random.choice(len(vols), size=n, p=vols / vols.sum())

    return np.einsum(
        "ijk, ij -> ik", deln[sample], dirichlet.rvs([1] * (dims + 1), size=n)
    )


def sample_from_convex_hull(
    z: Tensor,
    labels: Tensor,
    plot_dir: Path,
    target_labels: List[int],
    n: int = 5000,
    remove_outliers: bool = True,
    outlier_contamination: float = 0.25,
) -> Tuple[Tensor, Tensor]:
    """Compute the convex hull of each classes latents and sample from it.

    Args:
        z (Tensor): Latent tensors
        labels (Tensor): Ground truth labels
        plot_dir (Path): Directory to save plots in
        target_labels (List[int]): Labels to generate examples for
        n (int): Number of examples to create
        remove_outliers (bool): Remove outliers in latent space
        outlier_contamination (float): Rate by which outliers are removed

    Returns:
        Tuple[Tensor, Tensor]: latents, labels
    """
    generated_z = []
    generated_labels = []
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))

    # build convex hulls
    for i in target_labels:
        # get points that have label i
        mask = labels == i
        points = z[mask]
        # find outliers
        outlier_mask = (
            EllipticEnvelope(contamination=outlier_contamination).fit_predict(points)
            == 1
            if remove_outliers
            else True
        )
        # sample new latent variables from convex hull of points
        generated_z.append(
            dist_in_hull(points[outlier_mask], n=n // len(target_labels))
        )
        generated_labels.append([i] * (n // len(target_labels)))
        # plot points reduced to 2d, with 2d convex hull
        points_2d = (
            PCA(n_components=2).fit(points).transform(points)
            if points.shape[-1] > 2
            else points
        )
        ax.scatter(*points_2d.T, s=1, color=get_cmap("tab10")(i), label=i)
        hull_2d = ConvexHull(points_2d[outlier_mask])
        for simplex in hull_2d.simplices:
            ax.plot(
                *points_2d[outlier_mask][simplex].T, "k-", color=get_cmap("tab10")(i)
            )

    plt.legend()
    plt.savefig(plot_dir / "convex_hulls.png")
    plt.close()

    generated_z = Tensor(np.concatenate(generated_z, axis=0))
    generated_labels = torch.tensor(
        np.concatenate(generated_labels, axis=0), dtype=torch.int32
    )

    return generated_z, generated_labels


def generate_by_convex_hull(
    vae: VariationalAutoencoder,
    device: str,
    data_loader: DataLoader,
    plot_dir: Path,
    num_examples: int = 2000,
) -> Tuple[Tensor, Tensor]:
    """Generate new examples using convex hulls.

    Args:
        vae (VariationalAutoencoder): Variational autoencoder
        device (str): Device
        data_loader (DataLoader): Data loader from which examples are generated
        plot_dir (Path): Directory to save plots in
        num_examples (int): Number of examples to create

    Returns:
        Tuple[Tensor, Tensor]: generated examples, labels
    """
    # get means of encoded latent distributions
    with torch.no_grad():
        z_y = [(vae.encoder(x.to(device))[0].cpu(), y) for x, y in data_loader]
    z, labels = zip(*z_y)
    z, labels = (
        torch.cat(z, dim=0).numpy(),
        torch.cat(labels, dim=0).numpy(),
    )
    np.savetxt(plot_dir / "data" / "gt_mean.txt", z)
    np.savetxt(plot_dir / "data" / "gt_labels.txt", labels)

    # compute convex hull of means and sample from it
    z, labels = sample_from_convex_hull(
        z=z,
        labels=labels,
        plot_dir=plot_dir,
        target_labels=np.unique(labels),
        n=num_examples,
        remove_outliers=True,
    )
    np.savetxt(plot_dir / "data" / "gen_latents.txt", z)
    np.savetxt(plot_dir / "data" / "gen_labels.txt", labels)

    with torch.no_grad():
        x_ = vae.decoder(z.to(device)).cpu()

    # plot generated latents
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
    for i in np.unique(labels):
        mask = labels == i
        z_2d = PCA(n_components=2).fit(z).transform(z) if z.shape[-1] > 2 else z
        ax.scatter(*z_2d[mask].T, s=1, color=get_cmap("tab10")(i), label=i)
    plt.legend()
    plt.savefig(plot_dir / "generated_z.png")
    plt.close()

    # plot generated examples
    rows, cols = np.unique(labels).shape[0], 1
    sorted_x_ = []
    for i in np.unique(labels):
        mask = labels == i
        sorted_x_.append(x_[mask][:cols])
    sorted_x_ = torch.cat(sorted_x_, dim=0)[: rows * cols].numpy()
    plt.figure(figsize=(cols, rows))
    plt.imshow(reshape_to_img(sorted_x_, 28, 28, rows, cols), cmap="gray")
    plt.savefig(plot_dir / "generated_images.png")
    plt.close()

    with open(plot_dir / "data" / "gen_img.pkl", "wb") as f:
        pickle.dump(sorted_x_, f)

    return x_, labels


if __name__ == "__main__":
    plot_dir = (
        Path("./data/generation/MNIST/VAE")
        / f"z_dim={z_dim}_alpha={alpha}_beta={beta}_epochs={epochs}_"
        / "convex_hull"
    )
    plot_dir.mkdir(exist_ok=True, parents=True)
    (plot_dir / "data").mkdir(exist_ok=True, parents=True)
    generate_by_convex_hull(*model_setup(train=False), plot_dir, num_examples=200)
