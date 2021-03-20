"""CNN Model for MNIST."""
import torch.nn as nn
from torch import Tensor
from utils import init_weights

from utils.models import BaseModel, ModelConfig, ModelOutput


class CNNMNIST(BaseModel):
    """Convolutional Neural Network for classification."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        """Initialize the CNN.

        Args:
            config (ModelConfig): Model configuration
        """
        super(CNNMNIST, self).__init__(config)
        self.sequential = nn.Sequential(
            # input size: (nc) x 28 x 28
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (64) x 28 x 28
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (128) x 28 x 28
            nn.MaxPool2d(2),
            # state size: (128) x 14 x 14
            nn.Dropout(0.5),
            # ---------------------------
            # state size: (128) x 14 x 14
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (128) x 14 x 14
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (128) x 14 x 14
            nn.MaxPool2d(2),
            # state size: (128) x 7 x 7
            nn.Dropout(0.5),
            # ---------------------------
            # state size: (128) x 7 x 7
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # state size: 128
            nn.Linear(128, 10),
        )
        self.criterion = nn.CrossEntropyLoss()
        # init weights
        self.sequential.apply(init_weights)

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
