# noqa: D100
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader
from VAE.evaluation.evaluation_setup import eval_setup
from VAE.vae_model_v1 import VariationalAutoencoder


def visualize_feature_space(
    vae: VariationalAutoencoder, device: str, test_loader: DataLoader, eval_dir: Path
) -> None:
    """Visualize the feature space of the encoder.

    Args:
        vae: Variational autoencoder model
        device: Device
        test_loader: Test DataLoader
        eval_dir: Evaluation dir
    """
    filename = "feature_space.png"

    # get means of latent distribution and labels of each example
    with torch.no_grad():
        z_y = [(vae.encoder(x.to(device))[0].cpu(), y) for x, y in test_loader]
        z, test_labels = zip(*z_y)
        z, test_labels = (
            torch.cat(z, dim=0).numpy(),
            torch.cat(test_labels, dim=0).numpy(),
        )

    # perform PCA if latent space has dim > 2
    reduced_z = PCA(n_components=2).fit(z).transform(z) if vae.z_dim > 2 else z

    # plot latent means for each class
    plt.figure(figsize=(16, 16))
    for i in range(10):
        mask = test_labels == i
        plt.scatter(reduced_z[mask, 0], reduced_z[mask, 1], s=10)
    plt.legend(range(10))
    plt.savefig(eval_dir / filename)

    print(f"Feature space visualization results in {eval_dir / filename}")


if __name__ == "__main__":
    visualize_feature_space(*eval_setup())
