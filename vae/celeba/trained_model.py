"""Module loading."""
from typing import Tuple

import mlflow
import torch

from vae.celeba.model import VariationalAutoencoder


def get_model(
    epochs: int,
    z_dim: int,
    beta: float,
    cuda: bool = True,
) -> Tuple[Tuple[VariationalAutoencoder, str], mlflow.entities.Run]:
    """Loads the pytorch VAE model and mlflow run.

    Args:
        epochs (int): Number of epochs
        z_dim (int): Dimension of latent space
        beta (float): Beta
        cuda (bool): Use cuda if available

    Returns:
        Tuple[Tuple[VariationalAutoencoder, str], mlflow.entities.Run]: (VAE, device), mlrun
    """
    mlflow.set_tracking_uri("experiments/CelebA")
    experiment_id = mlflow.get_experiment_by_name("VAE").experiment_id
    filter_str = (
        f"params.EPOCHS = '{epochs}' AND "
        + f"params.Z_DIM = '{z_dim}' AND "
        + f"params.BETA = '{beta}'"
    )
    try:
        run = mlflow.search_runs(
            experiment_ids=[experiment_id], filter_string=filter_str
        ).iloc[0]
    except IndexError:
        raise LookupError()

    device = "cuda:0" if cuda and torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    return (
        mlflow.pytorch.load_model(run.artifact_uri + "/model").to(device),
        device,
    ), run


if __name__ == "__main__":
    (model, _), run = get_model(10, 100, 1.0)
    print(model)
    print(run)
