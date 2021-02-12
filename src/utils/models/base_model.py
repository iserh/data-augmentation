from .model_config import ModelConfig
from .model_output import ModelOutput
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super(BaseModel, self).__init__()
        self.config = config
        # initial default device is cpu
        self.config.device = self.config.device or "cpu"

    def to(self, device: str, *args, **kwargs) -> "BaseModel":
        self.config.device = device
        return super().to(device, *args, **kwargs)

    def forward(self) -> ModelOutput:
        raise NotImplementedError()

    def save(self, epochs: int) -> None:
        raise NotImplementedError()

    @staticmethod
    def from_pretrained(config: ModelConfig, epochs: int) -> "BaseModel":
        raise NotImplementedError()
