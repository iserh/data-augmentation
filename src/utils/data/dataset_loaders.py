from typing import Any, Union

import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CelebA, CIFAR10


def _get_mnist(train: bool = False) -> MNIST:
    return MNIST(
        root="~/torch_datasets",
        transform=transforms.ToTensor(),
        train=train,
        target_transform=torch.as_tensor,
    )


def _get_cifar10(train: bool = False) -> CIFAR10:
    return CIFAR10(
        root="~/torch_datasets",
        transform=transforms.ToTensor(),
        train=train,
        target_transform=torch.as_tensor,
    )


def _get_celeba(train: bool = False, target_type: str = "identity") -> CelebA:
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
        target_type=target_type,
        target_transform=torch.as_tensor,
    )


_datasets = {
    "MNIST": _get_mnist,
    "CelebA": _get_celeba,
    "CIFAR10": _get_cifar10,
}


def get_dataset(name: str, *args: Any, **kwargs: Any) -> Union[MNIST, CelebA]:
    return _datasets[name](*args, **kwargs)
