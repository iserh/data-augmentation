"""The Base Model class."""
from typing import Optional

import torch.nn as nn
from torch import Tensor

from .model_config import ModelConfig
from .model_output import ModelOutput


class BaseModel(nn.Module):
    """The Base Model class.

    Subclasses must implement:
    - forward
    - save
    - from_pretrained
    """

    def __init__(self, config: ModelConfig) -> None:
        """Stores the model config. Sets the default device to 'cpu' if not provided in config.

        Args:
            config (ModelConfig): Model configuration
        """
        super(BaseModel, self).__init__()
        self.config = config
        # initial default device is cpu
        self.config.device = self.config.device or "cpu"

    def to(self, device: str, *args, **kwargs) -> "BaseModel":
        """Move model to device. Calls pytorch's 'to' method but remembers the device it's moved to.

        Args:
            device (str): Device to move to
            *args: args for pytorch's 'to' method
            **kwargs: kwargs for pytorch's 'to' method

        Returns:
            BaseModel: Model on device
        """
        self.config.device = device
        return super().to(device, *args, **kwargs)

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> ModelOutput:
        """Implement the model forward method here.

        Args:
            x (Tensor): Input batch
            y (Optional[Tensor]): Label batch

        Raises:
            NotImplementedError: This method has to be implemented in subclass

        Returns:
            ModelOutput: The forward pass output
        """
        raise NotImplementedError()
        return

    def save(self, epochs: int) -> None:
        """Implement saving of the model here.

        Args:
            epochs (int): Number of epochs the model was trained

        Raises:
            NotImplementedError: This method has to be implemented in subclass
        """
        raise NotImplementedError()

    @staticmethod
    def from_pretrained(config: ModelConfig, epochs: int) -> "BaseModel":
        """Implement loading of a pretrained model here.

        Args:
            config (ModelConfig): Model configuration
            epochs (int): Number of epochs the model was trained

        Raises:
            NotImplementedError: This method has to be implemented in subclass

        Returns:
            BaseModel: The loaded model
        """
        raise NotImplementedError()
        return
