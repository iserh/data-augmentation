from typing import Callable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate


class BatchCollector:
    """TODO: better message: Always pass tensors to BatchCollector."""

    def __init__(self, apply_fn: Callable, k: int = 1) -> None:
        self.data = []  # collects all data the data
        self.apply = apply_fn

    def execute(self) -> Tuple[Tuple[Tensor, ...]]:
        # execute function on collected data and clear the collector
        stacked_y = self.apply(*(torch.stack(ts, dim=0) for ts in zip(*self.data)))
        self.data.clear()
        stacked_y = stacked_y if isinstance(stacked_y, tuple) else (stacked_y,)
        return tuple(torch.unbind(t, dim=0) for t in stacked_y)

    def __call__(self, *data: Tensor) -> "BatchCollector":
        self.data.append(data)
        return self

    @staticmethod
    def collate_fn(batch):  # noqa: ANN001
        # flatten out batch to process each element separately  # noqa: ANN205
        batch = np.asarray(batch, dtype=object)
        flat_batch = batch.reshape(-1)
        # TODO: is there a nice way to vectorize this?
        # get collectors
        collectors = {c for c in flat_batch if isinstance(c, BatchCollector)}
        collector_ys = {c: tuple(zip(*c.execute())) for c in collectors}
        collector_is = {c: 0 for c in collectors}
        # replace all collector instances by the correct value
        for i, v in enumerate(flat_batch):
            if v in collectors:
                idx = collector_is[v]
                flat_batch[i] = collector_ys[v][idx]
                collector_is[v] += 1
        # reshape batch and let pytorch handle the stacking
        batch = flat_batch.reshape(batch.shape)
        return default_collate(batch.tolist())
