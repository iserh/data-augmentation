from types import FunctionType
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset


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
            self.targets = classes[self.indices]
        else:
            self.indices = torch.randperm(len(dataset))[:n]
            self.targets = None

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

    def __getitem__(self, idx) -> Tuple[torch.Tensor]:
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


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    from utils.data.batch_collector import BatchCollector

    A, B = torch.arange(10), torch.arange(10, 20)
    dataA = TensorDataset(A, B)
    dataB = TensorDataset(B, A)

    # create a sample batch collector
    @BatchCollector
    def Sum(a, b):
        return a + b

    @BatchCollector
    def Sub(a, b):
        return a - b

    def sub2(a, b):
        return a - b

    # create a lambda dataset that uses the batch collector
    sum_data = LambdaDataset(dataA, Sum)
    sub_data = LambdaDataset(dataA, Sub)
    sub_data2 = LambdaDataset(dataA, sub2)
    # create a concatenated dataset
    direct_sum_data = TensorDataset(A + B)
    all_data = StackDataset(sum_data, direct_sum_data)
    # create a dataloader
    # make sure to use the batch collecor's collate_fn
    loader = DataLoader(all_data, batch_size=4, shuffle=True, collate_fn=BatchCollector.collate_fn, pin_memory=True)
    print(sub_data2[0])

    # resulting batches
    for batch in loader:
        print(batch)
