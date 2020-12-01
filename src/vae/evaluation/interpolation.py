# noqa: D100
from pathlib import Path

from data_augmentation.VAE.evaluation.evaluation_setup import eval_setup
from data_augmentation.VAE.vae_model_v1 import VariationalAutoencoder

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader


def visualize_grid_interpolation(
    vae: VariationalAutoencoder, device: str, test_loader: DataLoader, eval_dir: Path
) -> None:
    """Visualize the generated images from a [-2, 2] intervall in latent space.

    Args:
        vae: Variational autoencoder model
        device: Device
        test_loader: Test DataLoader
        eval_dir: Evaluation dir
    """
    filename = "interpolation_grid.png"
    n = 16

    z_plane = np.mgrid[(slice(-2, 2, 4 / n),) * 2]
    z_plane = np.concatenate((z_plane, np.zeros((vae.z_dim - 2, n, n))), axis=0)
    z_plane = z_plane.transpose(1, 2, 0).reshape(n * n, vae.z_dim).astype(np.float32)

    with torch.no_grad():
        x_ = vae.decoder(torch.from_numpy(z_plane).to(device)).cpu()

    x_ = x_.numpy().reshape(n, n, 28, 28).transpose(0, 2, 1, 3).reshape(n * 28, n * 28)

    plt.figure(figsize=(n, n))
    plt.imshow(x_, cmap="gray")
    plt.savefig(eval_dir / filename)
    plt.close()

    print(f"Grid interpolation results in {eval_dir / filename}")


def interpolate_two_ciffers(
    vae: VariationalAutoencoder, device: str, test_loader: DataLoader, eval_dir: Path
) -> None:
    """Interpolate between two ciffers of the dataset.

    Args:
        vae: Variational autoencoder model
        device: Device
        test_loader: Test DataLoader
        eval_dir: Evaluation dir
    """
    filename = "interpolation_two_ciffers.png"

    y1, y2 = 1, 7
    n = 10

    # get mean of latent distribution and label for each example
    means, labels = [], []
    for x, y in test_loader:
        with torch.no_grad():
            mean = vae.encoder(x.to(device))[0].cpu()
            means.append(mean)
            labels.append(y)

    # concat all values
    means = torch.cat(means)
    labels = torch.cat(labels)

    # get the means of the desired labels, create interpolation linspace for means
    z = torch.Tensor(np.linspace(means[(labels == y1)][0], means[(labels == y2)][0], n))

    # compute interpolated images
    with torch.no_grad():
        X_ = vae.decoder(z.to(device)).cpu()

    # reshape to output format
    X_ = X_.numpy().reshape(1, n, 28, 28).transpose(0, 2, 1, 3).reshape(1 * 28, n * 28)

    # plot
    plt.figure(figsize=(10, n))
    plt.imshow(X_, cmap="gray")
    plt.savefig(eval_dir / filename)

    print(f"Interpolation between two ciffers results in {eval_dir / filename}")


if __name__ == "__main__":
    setup_objects = eval_setup()
    visualize_grid_interpolation(*setup_objects)
    interpolate_two_ciffers(*setup_objects)
