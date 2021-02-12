"""Variational autoencoder module class."""
import torch.nn as nn

from utils import init_weights


class MLP(nn.Sequential):
    def __init__(self) -> None:
        super(MLP, self).__init__(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )
        self.code_paths = [__file__]

        # initialize weights
        self.apply(init_weights)
