"""Miscellaneous utilities."""
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    """Initialize the weights of a module.

    Args:
        module (nn.Module): The module
    """
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
    elif type(module) == nn.Conv2d:
        nn.init.dirac_(module.weight)
    elif type(module) == nn.ConvTranspose2d:
        nn.init.dirac_(module.weight)


def uri_to_path(uri: str) -> Path:
    """Convert an uri to a path.

    Args:
        uri (str): The uri to convert

    Returns:
        Path: The converted path
    """
    return Path(unquote(urlparse(uri).path))
