from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class TrainingArguments:
    """Training arguments dataclass."""

    epochs: Optional[int] = None
    no_cuda: bool = False
    save_epochs: Optional[int] = None
    save_model: bool = True
    log_steps: Optional[int] = 50
    seed: Optional[int] = None
    batch_size: int = 64
    metric_for_best_model: Optional[str] = None
    weight_decay: float = 0
    lr: float = 0.001
    num_workers: int = 4
    optimizer: Optional[Any] = None
