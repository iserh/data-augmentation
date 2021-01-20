from types import FunctionType
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class LambdaDataset(Dataset):
    """Dataset with a lambda function applied to each element, when calling __getitem__."""

    def __init__(self, dataset: Dataset, fn: FunctionType) -> None:
        self.dataset = dataset
        self.fn = fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        return self.fn(*self.dataset[idx])


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
