from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingArguments:
    epochs: int = 20
    no_cuda: bool = False
    save_intervall: Optional[int] = None
    seed: Optional[int] = None
