"""Module loading."""
from typing import List, Optional, Tuple

import mlflow
import torch.cuda as cuda

from vae.vae_model import VariationalAutoencoder


def load_model(
    epochs: int,
    z_dim: int,
    alpha: float,
    beta: float,
    target_labels: Optional[List[int]] = None,
    n_examples_limit: Optional[int] = None,
    use_cuda: bool = True,
) -> Tuple[Tuple[VariationalAutoencoder, str], mlflow.entities.Run]:
    """Loads the pytorch VAE model and mlflow run.

    Args:
        epochs (int): Number of epochs
        z_dim (int): Dimension of latent space
        alpha (float): Alpha
        beta (float): Beta
        target_labels (Optional[int]): Target labels, selects all if None
        n_examples_limit (Optional[List[int]]): Limit number of examples for training
        use_cuda (bool): Use cuda if available

    Returns:
        Tuple[Tuple[VariationalAutoencoder, str], mlflow.entities.Run]: (VAE, device), mlrun
    """
    experiment_id = mlflow.get_experiment_by_name("VAE MNIST").experiment_id
    filter_str = (
        f"params.EPOCHS = '{epochs}' AND "
        + f"params.Z_DIM = '{z_dim}' AND "
        + f"params.ALPHA = '{alpha}' AND "
        + f"params.BETA = '{beta}' AND "
        + f"params.TARGET_LABELS = '{target_labels}' AND "
        + f"params.N_EXAMPLES_LIMIT = '{n_examples_limit}'"
    )
    try:
        run = mlflow.search_runs(
            experiment_ids=[experiment_id], filter_string=filter_str
        ).iloc[0]
    except IndexError:
        raise LookupError()

    device = "cuda:0" if cuda.is_available() and use_cuda else "cpu"
    print("Using device:", device)

    return (
        mlflow.pytorch.load_model(run.artifact_uri + "/model").to(device),
        device,
    ), run


if __name__ == "__main__":
    (model, _), run = load_model(20, 2, 1.0, 1.0)
    print(model)
    print(run)
