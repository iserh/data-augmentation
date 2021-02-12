"""Standard model configuration dataclass."""
from dataclasses import dataclass, field
from typing import Optional, Any, Dict


@dataclass
class ModelConfig:
    device: Optional[str] = field(default=None, compare=False)
    attr: Optional[Dict[str, Any]] = None
