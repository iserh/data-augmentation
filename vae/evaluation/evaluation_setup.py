"""Evaluation."""
from pathlib import Path
from typing import Tuple

from torch.utils.data.dataloader import DataLoader

from utils import config
from vae.model_setup import alpha, beta, epochs, model_setup, z_dim
from vae.vae_model_v1 import VariationalAutoencoder


def eval_setup() -> Tuple[VariationalAutoencoder, str, DataLoader, Path]:
    """Setup for evaluation.

    Returns:
        VAE model, device, test DataLoader, evaluation dir path
    """
    eval_dir = Path(
        config.eval_path
        / f"MNIST/VAE/z_dim={z_dim}_alpha={alpha}_beta={beta}_epochs={epochs}"
    )
    eval_dir.mkdir(exist_ok=True, parents=True)
    vae, device, test_loader = model_setup()

    return vae, device, test_loader, eval_dir
