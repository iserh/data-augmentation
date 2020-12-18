"""Training script for variational autoencoder on mnist."""
import pickle
from pathlib import Path
from shutil import copyfile

from utils import config
from vae import vae_model_v1 as vae_model

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


def train_mnist(epochs: int, z_dim: int, alpha: float = 1.0, beta: float = 1.0) -> None:
    """Train variational autoencoder on mnist.

    Args:
        epochs: Number of epochs
        z_dim: Dimension of the latent space
        alpha: Alpha for loss
        beta: Beta for loss
    """
    # create model log directory
    log_dir = Path(
        config.model_path
        / f"MNIST/VAE/z_dim={z_dim}_alpha={alpha}_beta={beta}_epochs={epochs}"
    )
    log_dir.mkdir(exist_ok=True, parents=True)

    # Use cuda if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Model
    vae = vae_model.VariationalAutoencoder(z_dim=z_dim).to(device)
    # Optimizer
    optim = torch.optim.Adam(vae.parameters())
    # Loss
    vae_loss = vae_model.vae_loss
    # Save model architecture
    copyfile(Path(vae_model.__file__), log_dir / "vae_model.py")

    # Load datasets
    mnist_train = MNIST(
        root="~/torch_datasets",
        download=True,
        transform=transforms.ToTensor(),
        train=True,
    )
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

    # Losses
    bce_losses, kld_losses = [], []

    vae.train()
    for epoch in range(epochs):
        with tqdm(total=len(train_loader)) as pbar:
            pbar.set_description(f"Train Epoch {epoch + 1}/{epochs}", refresh=True)

            for x_true, _ in train_loader:
                x_true = x_true.to(device)

                # predict and compute loss
                x_hat, mean, variance_log = vae(x_true)
                bce_l, kld_l = vae_loss(x_hat, x_true, mean, variance_log, alpha, beta)

                # update parameters
                optim.zero_grad()
                (bce_l + kld_l).backward()
                optim.step()

                # update losses
                bce_losses.append(bce_l.item())
                kld_losses.append(kld_l.item())

                # progress
                pbar.set_postfix(
                    {"bce-loss": bce_losses[-1], "kld-loss": kld_losses[-1]}
                )
                pbar.update(1)

        # save model
        torch.save(vae.state_dict(), log_dir / "state_dict.pt")

        # save losses to file
        with open(log_dir / "losses.pkl", "wb") as outfile:
            pickle.dump({"bce_losses": bce_losses, "kld_losses": kld_losses}, outfile)

        # plot losses
        plt.figure(figsize=(16, 8))
        plt.plot(bce_losses, label="Binary Cross Entropy")
        plt.plot(kld_losses, label="KL Divergence")
        plt.legend()
        plt.title("Losses")
        plt.savefig(log_dir / "losses.png")
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAE Training.")
    parser.add_argument(
        "-e", "--epochs", type=int, default=20, help="Epochs model trained"
    )
    parser.add_argument(
        "-z", "--z_dim", type=int, default=2, help="Dimension latent space"
    )
    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="Alpha")
    parser.add_argument("-b", "--beta", type=float, default=1.0, help="Beta")
    args = parser.parse_args()

    epochs = args.epochs
    z_dim = args.z_dim
    alpha = args.alpha
    beta = args.beta

    print("Initialize VAE training.")
    print(f"z_dim={z_dim}_alpha={alpha}_beta={beta}_epochs={epochs}")
    train_mnist(epochs=epochs, z_dim=z_dim, alpha=alpha, beta=beta)
