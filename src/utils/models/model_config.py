"""Standard model configuration dataclass."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """Model configuration dataclass."""

    device: Optional[str] = field(default=None, compare=False)
    attr: Optional[Dict[str, Any]] = None
