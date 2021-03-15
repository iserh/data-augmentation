from typing import Any, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, KMNIST, MNIST, CelebA, Omniglot

from .proben1 import proben1_dataset_loaders


def _get_mnist(train: bool = False, download: bool = False) -> MNIST:
    return MNIST(
        root="~/torch_datasets",
        transform=transforms.ToTensor(),
        train=train,
        download=download,
        target_transform=torch.as_tensor,
    )


def _get_cifar10(train: bool = False, download: bool = False) -> CIFAR10:
    return CIFAR10(
        root="~/torch_datasets",
        transform=transforms.ToTensor(),
        train=train,
        download=download,
        target_transform=torch.as_tensor,
    )


def _get_celeba(train: bool = False, target_type: str = "identity", download: bool = False) -> CelebA:
    return CelebA(
        root="~/torch_datasets",
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        ),
        split="train" if train else "test",
        download=download,
        target_type=target_type,
        target_transform=torch.as_tensor,
    )


def _get_kmnist(train: bool = False, download: bool = False) -> MNIST:
    return KMNIST(
        root="~/torch_datasets",
        transform=transforms.ToTensor(),
        train=train,
        download=download,
        target_transform=torch.as_tensor,
    )


def _get_omniglot(train: bool = False, download: bool = False) -> Omniglot:
    return Omniglot(
        root="~/torch_datasets",
        transform=transforms.ToTensor(),
        background=True,
        download=download,
        target_transform=torch.as_tensor,
    )


_datasets = {
    "MNIST": _get_mnist,
    "CelebA": _get_celeba,
    "CIFAR10": _get_cifar10,
    "Omniglot": _get_omniglot,
    "KMNIST": _get_kmnist,
    **proben1_dataset_loaders,
}


def get_dataset(name: str, *args: Any, **kwargs: Any) -> Dataset:
    return _datasets[name](*args, **kwargs)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = get_dataset("MNIST", train=True, download=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    x, y = next(iter(dataloader))
    print(x.size())
    print(y.size(), f"max={y.max().item()}, min={y.min().item()}")
