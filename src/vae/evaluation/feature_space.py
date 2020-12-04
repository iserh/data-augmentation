# noqa: D100
from pathlib import Path

from vae.evaluation.evaluation_setup import eval_setup
from vae.vae_model_v1 import VariationalAutoencoder

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader


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
    colors = plt.get_cmap("tab10")

    # get means of latent distribution and labels of each example
    with torch.no_grad():
        z_y = [(vae.encoder(x.to(device))[0].cpu(), y) for x, y in test_loader]
        z, test_labels = zip(*z_y)
        z, test_labels = (
            torch.cat(z, dim=0).numpy(),
            torch.cat(test_labels, dim=0).numpy(),
        )

    # perform PCA if latent space has dim > 2
    z = PCA(n_components=3).fit(z).transform(z) if vae.z_dim > 3 else z

    # plot result
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection="3d" if vae.z_dim >= 3 else None)
    for i in range(10):
        # get vectors that match label i
        mask = test_labels == i
        points = z[mask]
        # remove outliers
        points = points[(EllipticEnvelope(contamination=0.5).fit_predict(points) == 1)]
        points = points[np.random.choice(points.shape[0], 100)]
        # plot these points
        ax.scatter(
            *[z[mask, i] for i in range(z.shape[1])], s=1, color=colors(i), label=i
        )
        # compute convex hull
        hull = ConvexHull(points)
        # plot convex hull
        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])
            ax.plot(
                *[points[simplex, i] for i in range(points.shape[1])],
                "k-",
                color=colors(i),
            )
    ax.legend()
    plt.savefig(eval_dir / filename)

    print(f"Feature space visualization results in {eval_dir / filename}")


if __name__ == "__main__":
    visualize_feature_space(*eval_setup())
