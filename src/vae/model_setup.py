"""Module loading."""
from typing import Optional, Tuple

from vae.vae_model import VariationalAutoencoder

import mlflow
from torch.cuda import is_available


def load_model(
    epochs: int,
    z_dim: int,
    alpha: float,
    beta: float,
    target_label: Optional[int] = None,
    cuda: bool = True,
) -> Tuple[Tuple[VariationalAutoencoder, str], mlflow.entities.Run]:
    """Loads the pytorch VAE model and mlflow run.

    Args:
        epochs (int): Number of epochs
        z_dim (int): Dimension of latent space
        alpha (float): Alpha
        beta (float): Beta
        cuda (bool): Use cuda if available

    Returns:
        Tuple[Tuple[VariationalAutoencoder, str], mlflow.entities.Run]: (VAE, device), mlrun
    """
    experiment_id = mlflow.get_experiment_by_name("VAE MNIST").experiment_id
    filter_str = (
        f"params.epochs = '{epochs}' AND "
        + f"params.z_dim = '{z_dim}' AND "
        + f"params.alpha = '{alpha}' AND "
        + f"params.beta = '{beta}' AND "
        + f"params.target_label = '{target_label}'"
    )
    run = mlflow.search_runs(
        experiment_ids=[experiment_id], filter_string=filter_str
    ).iloc[0]

    device = "cuda:0" if is_available() and cuda else "cpu"
    print("Using device:", device)

    return (
        mlflow.pytorch.load_model(run.artifact_uri + "/model").to(device),
        device,
    ), run


if __name__ == "__main__":
    (model, _), run = load_model(20, 2, 1.0, 1.0)
    print(model)
    print(run)
