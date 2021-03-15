"""CNN Model for MNIST."""
import torch.nn as nn
from torch import Tensor

from utils.models import BaseModel, ModelConfig, ModelOutput


class ModelProben1(BaseModel):
    """Convolutional Neural Network for classification."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        """Initialize the CNN.

        Args:
            config (ModelConfig): Model configuration
        """
        super(ModelProben1, self).__init__(config)
        in_feat = config.attr["in_feat"]
        out_feat = config.attr["out_feat"]
        N = config.attr["N"]
        M = config.attr["M"]
        K = config.attr["K"]
        self.sequential = nn.Sequential(
            # input size: in_feat
            nn.Linear(in_feat, N),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(N),
            nn.Dropout(0.1, inplace=True),
            # state size: N
            nn.Linear(N, M),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(M),
            nn.Dropout(0.1, inplace=True),
            # state size: M
            nn.Linear(M, K),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(K),
            nn.Dropout(0.1, inplace=True),
            # state size: K
            nn.Linear(K, out_feat),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, y: Tensor) -> ModelOutput:
        """Forward pass.

        Args:
            x (Tensor): Input batch
            y (Tensor): Label batch

        Returns:
            ModelOutput: Output of the model
        """
        pred: Tensor = self.sequential(x)
        loss = self.criterion(pred, y)
        return ModelOutput(loss=loss, prediction=pred.argmax(1))
