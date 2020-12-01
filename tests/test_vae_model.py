"""Testing the VAE model."""
from src.data_augmentation.VAE.vae_model_v1 import VariationalAutoencoder

import torch


def test_model() -> None:
    """Test shapes of the model."""
    vae = VariationalAutoencoder(2)
    # create random input image
    x = torch.empty(size=(1, 1, 28, 28)).normal_(0, 1)
    with torch.no_grad():
        x_, _, _ = vae(x)

    assert x_.size() == (1, 1, 28, 28)
