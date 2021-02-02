"""Pretrained model config."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PretrainedConfig:
    epochs: int
    checkpoint: int
