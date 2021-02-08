"""Pretrained model config."""
from dataclasses import dataclass


@dataclass
class PretrainedConfig:
    epochs: int = 10
    checkpoint: int = epochs
