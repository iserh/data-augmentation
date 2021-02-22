"""Train a model."""
from typing import Dict, Optional

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from torch.utils.data import Dataset

from utils.models import BaseModel
from utils.trainer import Trainer, TrainingArguments


def train_model(
    model: BaseModel,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    test_dataset: Dataset,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Train a convolutional neural network on a dataset.

    Args:
        model (BaseModel): The model to train
        training_args (TrainingArguments): Arguments for the trainer
        train_dataset (Dataset): Dataset used for training
        dev_dataset (Dataset): Dataset used for test steps while training (early stopping)
        test_dataset (Dataset): Dataset used for evaluating the model
        seed (Optional[int]): Seed for reproducibility

    Returns:
        Dict[str, float]: Evaluation results
    """
    # seeding
    if seed is not None:
        torch.manual_seed(seed)
    # trainer
    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        epoch_metrics=BestTracker(),
    )
    # train model
    trainer.train()
    # evaluate model
    return trainer.evaluate()


class BestTracker:

    def __init__(self) -> None:
        self.best_acc = 0
    
    def __call__(self, predictions: Tensor, labels: Tensor, train: bool) -> Dict[str, float]:
        """Metrics for the trainer class, that are computed each train/eval step.

        Args:
            predictions (Tensor): The predictions batch
            labels (Tensor): The true label batch

        Returns:
            Dict[str, float]: Computed metrics
        """
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        if (not train) and acc > self.best_acc:
            self.best_acc = acc
            log_best = {"best_acc": acc}
        else:
            log_best = {}
        return {
            "acc": acc,
            "f1": f1,
            **log_best
        }
