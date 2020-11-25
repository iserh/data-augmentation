# noqa: D100
from pathlib import Path

from data_augmentation.VAE.evaluation.evaluation_setup import eval_setup
from data_augmentation.VAE.vae_model_v1 import VariationalAutoencoder

import torch
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader


def generate_from_test(
    vae: VariationalAutoencoder, device: str, test_loader: DataLoader, eval_dir: Path
) -> None:
    """Generate images by encoding - decoding the test dataset.

    Args:
        vae: Variational autoencoder model
        device: Device
        test_loader: Test DataLoader
        eval_dir: Evaluation dir
    """
    filenames = ["generated_from_test_original.png", "generated_from_test_gen.png"]

    n, m = 10, 20

    # get original image x and generated image x_ for each example
    X, X_ = [], []
    for x, _ in test_loader:
        with torch.no_grad():
            x_ = vae.forward(x.to(device))[0].cpu()
            X.append(x)
            X_.append(x_)

    # concat all images and reshape to fit plotting format
    X = (
        torch.cat(X)[: n * m]
        .numpy()
        .reshape(n, m, 28, 28)
        .transpose(0, 2, 1, 3)
        .reshape(n * 28, m * 28)
    )
    X_ = (
        torch.cat(X_)[: n * m]
        .numpy()
        .reshape(n, m, 28, 28)
        .transpose(0, 2, 1, 3)
        .reshape(n * 28, m * 28)
    )

    # plot original images
    plt.figure(figsize=(n, m))
    plt.imshow(X, cmap="gray")
    plt.savefig(eval_dir / filenames[0])

    # plot generated images
    plt.figure(figsize=(n, m))
    plt.imshow(X_, cmap="gray")
    plt.savefig(eval_dir / filenames[1])

    print(
        f"Generated images from test results in {eval_dir / filenames[0]}"
        + f" and {eval_dir / filenames[1]}"
    )


def generate_from_normal(
    vae: VariationalAutoencoder, device: str, test_loader: DataLoader, eval_dir: Path
) -> None:
    """Generate images by decoding random samples drawn from a normal distribution.

    Args:
        vae: Variational autoencoder model
        device: Device
        test_loader: Test DataLoader
        eval_dir: Evaluation dir
    """
    filename = "generated_from_normal.png"

    n, m = 10, 20

    # create random latent vectors
    z = torch.empty((n * m, vae.z_dim)).normal_()

    # generate images from random Z vectors
    with torch.no_grad():
        X_ = vae.decoder(z.to(device)).cpu()

    # reshape to output format
    X_ = X_.numpy().reshape(n, m, 28, 28).transpose(0, 2, 1, 3).reshape(n * 28, m * 28)

    # plot
    plt.figure(figsize=(n, m))
    plt.imshow(X_, cmap="gray")
    plt.show()
    plt.savefig(eval_dir / filename)

    print(f"Generated images from normal results in {eval_dir / filename}")


if __name__ == "__main__":
    setup_objects = eval_setup()
    generate_from_test(*setup_objects)
    generate_from_normal(*setup_objects)
