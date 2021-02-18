from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingArguments:
    """Training arguments dataclass."""

    epochs: int = 20
    no_cuda: bool = False
    save_intervall: Optional[int] = None
    save_model: bool = True
    seed: Optional[int] = None
