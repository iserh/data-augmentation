from typing import Optional
import torch
from torch.utils.data import random_split
from utils.data import get_dataset
from pathlib import Path


def create_all_datasets(dataset_name: str, dataset_limit: int, seed: Optional[int] = None):
    # seed torch
    if seed is not None:
        torch.manual_seed(seed)

    # load train dataset
    train_dataset = get_dataset(dataset_name, train=True)
    # load test dataset
    test_dataset = get_dataset(dataset_name, train=False)
    # limit train dataset corresponding to DATASET_LIMIT and add 5000 for dev dataset
    vae_train_size = round(0.85 * len(train_dataset))
    vae_train_dataset, val_dataset = random_split(
        train_dataset, [vae_train_size, len(train_dataset) - vae_train_size]
    )
    train_dataset, _ = random_split(train_dataset, [dataset_limit, len(train_dataset) - dataset_limit])

    path = Path(f"datasets/{dataset_name}")
    path.mkdir(parents=True, exist_ok=True)

    torch.save(train_dataset, path / "train.pt")
    torch.save(vae_train_dataset, path / "vae_train.pt")
    torch.save(val_dataset, path / "val.pt")
    torch.save(test_dataset, path / "test.pt")


def create_train_dataset(dataset_name: str, dataset_limit: int, seed: Optional[int] = None):
    # seed torch
    if seed is not None:
        torch.manual_seed(seed)

    # load train dataset
    train_dataset = get_dataset(dataset_name, train=True)
    train_dataset, _ = random_split(train_dataset, [dataset_limit, len(train_dataset) - dataset_limit])

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
