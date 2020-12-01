# noqa: D100
from pathlib import Path

from vae.evaluation.evaluation_setup import eval_setup
from vae.vae_model_v1 import VariationalAutoencoder

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader


def plot_single_distribution(
    vae: VariationalAutoencoder, device: str, test_loader: DataLoader, eval_dir: Path
) -> None:
    """Generate images by encoding - decoding the test dataset.

    Args:
        vae: Variational autoencoder model
        device: Device
        test_loader: Test DataLoader
        eval_dir: Evaluation dir
    """
    filename = "single_distribution.png"

    target = 4
    m = 1

    # get mean of latent distribution and label for each example
    means, variances, labels = [], [], []
    for x, y in test_loader:
        with torch.no_grad():
            mean, log_variance = vae.encoder(x)
            means.append(mean.cpu())
            variances.append(log_variance.exp().cpu())
            labels.append(y)

    # concat all values
    means = torch.cat(means)
    variances = torch.cat(variances)
    labels = torch.cat(labels)

    mean1, variance1 = means[(labels == target)][:m], variances[(labels == target)][:m]

    samples = np.random.normal(
        mean1.numpy(), variance1.numpy(), size=(100, m, vae.z_dim)
    )
    samples = np.concatenate(samples, axis=0)

    reduced_samples = (
        PCA(n_components=2).fit(samples).transform(samples)
        if vae.z_dim > 2
        else samples
    )

    plt.figure(figsize=(10, 10))
    sns.kdeplot(
        reduced_samples[:, 0],
        y=reduced_samples[:, 1],
        fill=True,
        clip=((-4, 4), (-4, 4)),
        label=target,
    )
    plt.legend()
    plt.savefig(eval_dir / filename)


if __name__ == "__main__":
    plot_single_distribution(*eval_setup())
