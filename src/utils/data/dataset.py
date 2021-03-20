from types import FunctionType
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset


class BatchDataset(Dataset):
    def __init__(self, dataset: Dataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
    
    def dataset_length(self) -> int:
        return len(self.dataset)

    def __len__(self) -> int:
        return max(len(self.dataset), self.batch_size)

    def __getitem__(self, idx: int) -> Any:
        idx = idx % len(self.dataset)
        return self.dataset[idx]


class LambdaDataset(Dataset):
    """Dataset with a lambda function applied to each element, when calling __getitem__."""

    def __init__(self, dataset: Dataset, fn: FunctionType, k: int = 1) -> None:
        self.dataset = dataset
        self.fn = fn
        self.k = k

    def __len__(self) -> int:
        return len(self.dataset) * self.k

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        return self.fn(*self.dataset[idx // self.k])


class ResizeDataset(Dataset):
    """Dataset that limits the amount of data accessible."""

    def __init__(self, dataset: Dataset, n: int, classes: Optional[Tensor] = None) -> None:
        self.dataset = dataset
        if classes is not None:
            class_indices = [torch.where(classes == c)[0] for c in torch.unique(classes)]
            assert n % len(class_indices) == 0, "Size of dataset must be a multiple of the unique class count."
            class_indices = [indices[: n // len(class_indices)] for indices in class_indices]
            self.indices = torch.cat(class_indices, dim=0)
            self.labels = classes[self.indices]
        else:
            self.indices = torch.randperm(len(dataset))[:n]
            self.labels = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        return self.dataset[self.indices[idx]]


class LoaderDataset(torch.utils.data.Dataset):
    """Create a dataset from a dataloader.

    Dataloaders for this dataset cannot use multiprocessing!
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader) -> None:
        self.loader = dataloader
        self._iter = iter(self.loader)
        self._i, self._batch = 0, None

    def __getitem__(self, idx) -> Tuple[torch.Tensor]:  # noqa: ANN001
        if (self._batch is None) or (self._i == self._batch[0].size(0)):
            try:
                self._batch = next(self._iter)
            except StopIteration:
                self._iter = iter(self.loader)
            self._i = 0
        data = tuple(t[self._i] for t in self._batch)
        self._i += 1
        return data

    def __len__(self) -> int:
        return len(self.loader.dataset)

    def get_tensors(self) -> Tuple[Tensor, ...]:
        return next(iter(DataLoader(self, batch_size=len(self))))

    def load(self) -> TensorDataset:
        return TensorDataset(*next(iter(DataLoader(self, batch_size=len(self)))))


class DataFetcher:
    def __init__(
        self, dataset: Dataset, n_samples: int, batch_size: int = 512, num_workers: int = 0, shuffle: bool = False
    ) -> None:
        loader_dataset = LoaderDataset(
            DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        )
        self.dataloader = DataLoader(loader_dataset, batch_size=n_samples)

    def fetch(self) -> TensorDataset:
        return TensorDataset(*next(iter(self.dataloader)))

    def __call__(self) -> TensorDataset:
        return self.fetch()
