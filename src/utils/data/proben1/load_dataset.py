from typing import Callable
import torch
from torch.utils.data import TensorDataset
from .create_tensordataset import pt_path


def load_proben1(dataset_name: str, train: bool = False, download: bool = False) -> TensorDataset:
    return torch.load(pt_path / f"{dataset_name}-{'train' if train else 'test'}.pt")
