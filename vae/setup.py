"""Config for training, analysing and generating."""
from typing import Optional, Union

from vae.celeba.dataset import CelebALoader
from vae.celeba.model import VariationalAutoencoder as CelebAVAE
from vae.mnist.dataset import MNISTLoader
from vae.mnist.model import VariationalAutoencoder as MNISTVAE


def get_model(dataset: str, z_dim: int, n_channels: Optional[int] = None) -> Union[MNISTVAE, CelebAVAE]:
    if dataset == "MNIST":
        return MNISTVAE(z_dim, n_channels or 1)
    elif dataset == "CelebA":
        return CelebAVAE(z_dim, n_channels or 3)
    else:
        raise KeyError(f"No model found for key '{dataset}'")


def get_dataloader(
    dataset: str,
    train: bool = False,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    pin_memory: Optional[bool] = None,
    target_type: Optional[str] = None,
) -> Union[MNISTLoader, CelebALoader]:
    if dataset == "MNIST":
        if train:
            return MNISTLoader(batch_size=batch_size or 128, shuffle=shuffle or True, train=True, pin_memory=True)
        else:
            return MNISTLoader(batch_size=batch_size or 512, shuffle=shuffle or False, train=False, pin_memory=True)
    elif dataset == "CelebA":
        if train:
            return CelebALoader(
                batch_size=batch_size or 128,
                shuffle=shuffle or True,
                train=True,
                target_type=target_type or "identity",
                pin_memory=pin_memory or True,
            )
        else:
            return CelebALoader(
                batch_size=batch_size or 512,
                shuffle=shuffle or False,
                train=False,
                target_type=target_type or "identity",
                pin_memory=pin_memory or True,
            )
    else:
        raise KeyError(f"No dataloader found for key '{dataset}'")
