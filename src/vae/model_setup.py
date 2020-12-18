"""Module loading."""
import argparse
import importlib.util
from pathlib import Path
from typing import Tuple

from utils import config
from vae.vae_model_v1 import VariationalAutoencoder

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser(description="VAE Model.")
parser.add_argument("-e", "--epochs", type=int, default=20, help="Epochs model trained")
parser.add_argument("-z", "--z_dim", type=int, default=2, help="Dimension latent space")
parser.add_argument("-a", "--alpha", type=float, default=1.0, help="Alpha")
parser.add_argument("-b", "--beta", type=float, default=1.0, help="Beta")
parser.add_argument("-c", "--cuda", type=bool, default=True, help="Use cuda")
args = parser.parse_args()

epochs: int = args.epochs
z_dim: int = args.z_dim
alpha: float = args.alpha
beta: float = args.beta
use_cuda: bool = args.cuda


def model_setup(
    train: bool = False,
) -> Tuple[VariationalAutoencoder, str, DataLoader, Path]:
    """Setup for evaluation.

    Args:
        train (bool): Use training dataset

    Returns:
        VAE model, device, test DataLoader
    """
    vae, device = load_vae_model(use_cuda)

    # Load dataset
    mnist_test = MNIST(
        root="~/torch_datasets",
        download=True,
        transform=ToTensor(),
        train=train,
    )
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)

    return vae, device, test_loader


def load_vae_model(use_cuda: bool) -> Tuple[VariationalAutoencoder, str]:
    """Load vae model.

    Args:
        use_cuda: Use cuda device if available

    Returns:
        vae model, device
    """
    log_dir = Path(
        config.model_path
        / f"MNIST/VAE/z_dim={z_dim}_alpha={alpha}_beta={beta}_epochs={epochs}"
    )

    # Use cuda if available
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"
    print("Using device:", device)

    # load model class
    spec = importlib.util.spec_from_file_location("vae_model", log_dir / "vae_model.py")
    vae_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vae_model)
    vae = vae_model.VariationalAutoencoder(z_dim).to(device)
    vae.load_state_dict(
        torch.load(log_dir / "state_dict.pt", map_location=torch.device(device))
    )

    return vae, device


if __name__ == "__main__":
    print(f"z_dim={z_dim}_alpha={alpha}_beta={beta}_epochs={epochs}")
