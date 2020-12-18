"""Data generation."""
import pickle
from pathlib import Path
from typing import Tuple

from utils.plotting import reshape_to_img
from vae.model_setup import alpha, beta, epochs, model_setup, z_dim
from vae.vae_model_v1 import VariationalAutoencoder

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import get_cmap
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data.dataloader import DataLoader


def generate_by_dist(
    vae: VariationalAutoencoder,
    device: str,
    data_loader: DataLoader,
    plot_dir: Path,
    num_examples: int = 2000,
) -> Tuple[Tensor, Tensor]:
    """Generate new examples using normal distributions.

    Args:
        vae (VariationalAutoencoder): Variational autoencoder
        device (str): Device
        data_loader (DataLoader): Data loader from which examples are generated
        plot_dir (Path): Directory to save plots in
        num_examples (int): Number of examples to create

    Returns:
        Tuple[Tensor, Tensor]: generated examples, labels
    """
    # extract dataloader
    x_y = [(x, y) for x, y in data_loader]
    x, labels = zip(*x_y)
    x, labels = (
        torch.cat(x, dim=0),
        torch.cat(labels, dim=0),
    )
    # run encoder stage
    with torch.no_grad():
        mv_y = [(vae.encoder(x.to(device)), y) for x, y in data_loader]
    # extract means, variance_logs, labels
    mv, labels = zip(*mv_y)
    mean, variance_log = zip(*mv)
    # concatenate batches
    mean, variance_log, labels = (
        torch.cat(mean, dim=0).cpu(),
        torch.cat(variance_log, dim=0).cpu(),
        torch.cat(labels, dim=0).cpu(),
    )
    # plot means
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
    for i in np.unique(labels.numpy()):
        mask = labels == i
        z_2d = (
            PCA(n_components=2).fit(mean).transform(mean)
            if mean.shape[-1] > 2
            else mean
        )
        ax.scatter(*z_2d[mask].T, s=1, color=get_cmap("tab10")(i), label=i)
    plt.legend()
    plt.savefig(plot_dir / "gt_latents.png")
    plt.close()

    np.savetxt(plot_dir / "data" / "gt_mean.txt", mean)
    np.savetxt(plot_dir / "data" / "gt_var_log.txt", variance_log)
    np.savetxt(plot_dir / "data" / "gt_labels.txt", labels)

    # draw random examples from dataset
    rand_indices = torch.randint(0, x.shape[0], (num_examples,))
    mean, variance_log = mean[rand_indices], variance_log[rand_indices]
    labels = labels[rand_indices]

    # generate latents by sampling from encoder distributions
    eps = torch.empty_like(variance_log).normal_()
    z = eps * (0.5 * variance_log).exp() + mean

    # run decoder stage
    with torch.no_grad():
        x_ = vae.decoder(z.to(device)).cpu()

    # plot generated latents
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
    for i in np.unique(labels.numpy()):
        mask = labels == i
        z_2d = PCA(n_components=2).fit(z).transform(z) if z.shape[-1] > 2 else z
        ax.scatter(*z_2d[mask].T, s=1, color=get_cmap("tab10")(i), label=i)
    plt.legend()
    plt.savefig(plot_dir / "generated_z.png")
    plt.close()

    np.savetxt(plot_dir / "data" / "gen_latents.txt", z)
    np.savetxt(plot_dir / "data" / "gen_labels.txt", labels)

    # plot generated examples
    rows, cols = np.unique(labels.numpy()).shape[0], 1
    sorted_x_ = []
    for i in np.unique(labels.numpy()):
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
        / "distribution"
    )
    plot_dir.mkdir(exist_ok=True, parents=True)
    (plot_dir / "data").mkdir(exist_ok=True, parents=True)
    generate_by_dist(*model_setup(train=False), plot_dir, num_examples=20)
