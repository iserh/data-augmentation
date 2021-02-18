"""Output dataclass of a model's forward method."""
from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class ModelOutput:
    """Model output dataclass."""

    prediction: Optional[Tensor] = None
    loss: Optional[Tensor] = None
