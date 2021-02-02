"""Variational autoencoder module class."""
import torch.nn as nn
from torch import Tensor
from utils import init_weights


class MLP(nn.Module):

    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.code_paths = [__file__]
        self.linear_stage = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

        # initialize weights
        self.linear_stage.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_stage(x)
