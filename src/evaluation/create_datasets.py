from typing import Optional
import torch
from torch.utils.data import random_split
from utils.data import get_dataset

DATASET = "MNIST"
DATASET_LIMIT = 50


def create_datasets(seed: Optional[int] = None):
    # seed torch
    if seed is not None:
        torch.manual_seed(seed)

    # load train dataset
    train_dataset = get_dataset(DATASET, train=True)
    # load test dataset
    test_dataset = get_dataset(DATASET, train=False)
    # limit train dataset corresponding to DATASET_LIMIT and add 5000 for dev dataset
    vae_train_size = round(0.85 * len(train_dataset))
    vae_train_dataset, val_dataset = random_split(
        train_dataset, [vae_train_size, len(train_dataset) - vae_train_size]
    )
    train_dataset, _ = random_split(train_dataset, [DATASET_LIMIT, len(train_dataset) - DATASET_LIMIT])

    torch.save(train_dataset, f"datasets/{DATASET}/train.pt")
    torch.save(vae_train_dataset, f"datasets/{DATASET}/vae_train.pt")
    torch.save(val_dataset, f"datasets/{DATASET}/val.pt")
    torch.save(test_dataset, f"datasets/{DATASET}/test.pt")


def create_train_dataset(seed: Optional[int] = None):
    # seed torch
    if seed is not None:
        torch.manual_seed(seed)

    # load train dataset
    train_dataset = get_dataset(DATASET, train=True)
    train_dataset, _ = random_split(train_dataset, [DATASET_LIMIT, len(train_dataset) - DATASET_LIMIT])
    torch.save(train_dataset, f"datasets/{DATASET}/train.pt")


def load_datasets():
    train_dataset = torch.load(f"datasets/{DATASET}/train.pt")
    vae_train_dataset = torch.load(f"datasets/{DATASET}/vae_train.pt")
    val_dataset = torch.load(f"datasets/{DATASET}/val.pt")
    test_dataset = torch.load(f"datasets/{DATASET}/test.pt")
    return train_dataset, vae_train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # create_datasets(1337)
    create_train_dataset(1337)
    train_dataset, vae_train_dataset, val_dataset, test_dataset = load_datasets()
    print(len(train_dataset), len(vae_train_dataset), len(val_dataset), len(test_dataset))
