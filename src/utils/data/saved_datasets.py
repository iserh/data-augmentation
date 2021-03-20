from pathlib import Path
from typing import Optional, Tuple

import torch
from utils import seed_everything
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import TensorDataset

from utils.data import get_dataset


def create_all_datasets(dataset_name: str, dataset_limit: int, balance: bool = False, seed: Optional[int] = None):
    seed_everything(seed)

    # load train dataset
    train_dataset = get_dataset(dataset_name, train=True)
    # load test dataset
    test_dataset = get_dataset(dataset_name, train=False)
    # limit train dataset corresponding to DATASET_LIMIT and add 5000 for dev dataset
    vae_train_size = round(0.85 * len(train_dataset))
    vae_train_dataset, val_dataset = random_split(train_dataset, [vae_train_size, len(train_dataset) - vae_train_size])
    train_dataset, _ = random_split(train_dataset, [dataset_limit, len(train_dataset) - dataset_limit])

    path = Path(f"datasets/{dataset_name}")
    path.mkdir(parents=True, exist_ok=True)

    torch.save(train_dataset, path / "train.pt")
    torch.save(vae_train_dataset, path / "vae_train.pt")
    torch.save(val_dataset, path / "val.pt")
    torch.save(test_dataset, path / "test.pt")


def create_train_dataset(dataset_name: str, dataset_limit: int, balance: bool = False, seed: Optional[int] = None):
    seed_everything(seed)

    # load train dataset
    train_dataset = get_dataset(dataset_name, train=True)
    if not balance:
        train_dataset, _ = random_split(train_dataset, [dataset_limit, len(train_dataset) - dataset_limit])
    else:
        data, labels = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))
        classes = torch.unique(labels, sorted=True)
        n = dataset_limit // len(classes)
        data = torch.cat([data[labels == cls][:n] for cls in classes], dim=0)
        labels = torch.cat([torch.tensor([cls] * n, dtype=labels.dtype) for cls in classes], dim=0)
        train_dataset = TensorDataset(data, labels)

    path = Path(f"datasets/{dataset_name}")
    path.mkdir(parents=True, exist_ok=True)
    torch.save(train_dataset, path / "train.pt")


def load_datasets(dataset_name: str):
    path = Path(f"datasets/{dataset_name}")
    train_dataset = torch.load(path / "train.pt")
    vae_train_dataset = torch.load(path / "vae_train.pt")
    val_dataset = torch.load(path / "val.pt")
    test_dataset = torch.load(path / "test.pt")
    return train_dataset, vae_train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    create_train_dataset("MNIST", 50, balance=True, seed=1337)
    train_dataset, _, _, _ = load_datasets("MNIST")
    print(len(train_dataset))
    data, labels = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))
    classes, class_counts = torch.unique(labels, sorted=True, return_counts=True)
    print(classes)
    print(class_counts)
