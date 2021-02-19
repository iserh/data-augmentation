from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingArguments:
    """Training arguments dataclass."""

    epochs: Optional[int] = None
    total_steps: Optional[int] = None
    no_cuda: bool = False
    save_intervall: Optional[int] = None
    save_model: bool = True
    seed: Optional[int] = None
    batch_size: int = 128
    early_stopping: bool = False
    early_stopping_window: int = 10
    validation_intervall: Optional[int] = None
