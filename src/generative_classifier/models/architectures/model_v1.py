"""Variational autoencoder module class."""
import torch.nn as nn

from utils import init_weights
from utils.models import ModelConfig

from generative_classifier.models import GenerativeClassifierModel


class GenerativeClassifierV1(GenerativeClassifierModel):
    def __init__(self, config: ModelConfig) -> None:
        super(GenerativeClassifierV1, self).__init__(config)
        nc = 1
        self.model = nn.Sequential(
            # input size: (nc) x 28 x 28
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(nc, nc, 2, 1),
            nn.ReLU(inplace=True),
            # state size: (nc) x 28 x 28
            nn.Conv2d(nc, 64, 2, 2),
            nn.ReLU(inplace=True),
            # state size: (64) x 14 x 14
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (64) x 14 x 14
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (64) x 14 x 14
            nn.Flatten(),
            # state size: 64 * 14 * 14
            nn.Linear(64 * 14 * 14, 1)
            # output size: 1
        )

        # initialize weights
        self.model.apply(init_weights)


def _get_model_constructor() -> GenerativeClassifierV1:
    return GenerativeClassifierV1
