"""Output dataclass of a model's forward method."""
from dataclasses import dataclass

from torch import Tensor


@dataclass
class ModelOutput:
    loss: Tensor
