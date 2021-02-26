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
    """Metric Tracker, that remembers the best accuracy in validation."""

    def __init__(self) -> None:
        """Initialize BestTracker"""
        self.best_acc = 0

    def __call__(self, predictions: Tensor, labels: Tensor, validate: bool) -> Dict[str, float]:
        """Metrics for the trainer class, that are computed each train/eval step.

        Args:
            predictions (Tensor): The predictions batch
            labels (Tensor): The true label batch
            validate (bool): Indicates, that this is an validation step, so best is computed

        Returns:
            Dict[str, float]: Computed metrics
        """
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        if validate and acc > self.best_acc:
            self.best_acc = acc
            log_best = {"best_acc": acc}
        else:
            log_best = {}
        return {"acc": acc, "f1": f1, **log_best}


if __name__ == "__main__":
    import mlflow
    import torch
    import vae
    from utils.mlflow import backend_stores
    from utils.trainer import TrainingArguments
    from vae.generation import Generator, augmentations
    from vae.models import VAEConfig
    from evaluation.models import CNNMNIST
    from evaluation.train_utils import train_model
    from utils.data import load_datasets

    vae.models.base.model_store = "pretrained_models/MNIST"

    # *** Seeding, loading data & setting up mlflow logging ***

    SEED = 1337
    DATASET = "MNIST"
    # set the backend store uri of mlflow
    mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
    # seed torch
    torch.manual_seed(SEED)
    # load datasets
    train_dataset, _, val_dataset, test_dataset = load_datasets(DATASET)

    # *** The parameters for the classification task ***

    training_args = TrainingArguments(
        total_steps=5000,
        batch_size=32,
        validation_intervall=200,
        save_model=False,
        seed=SEED,
        early_stopping=False,
        early_stopping_window=20,
        save_best_metric="best_acc",
    )

    # *** Parameters for data augmentation ***

    MULTI_VAE = False
    VAE_EPOCHS = 50
    Z_DIM = 10
    BETA = 1.0
    AUGMENTATION = augmentations.DISTRIBUTION
    augmentation_params = {"K": 450}

    # *** Training the CNN ***

    # create a vae config
    vae_config = VAEConfig(z_dim=Z_DIM, beta=BETA)
    # start mlflow run in experiment
    mlflow.set_experiment(f"CNN Z_DIM {Z_DIM}" if AUGMENTATION else "CNN Baseline")
    with mlflow.start_run(run_name=AUGMENTATION or "baseline"):
        # perform data augmentation if specified
        if AUGMENTATION is not None:
            gen = Generator(
                vae_config,
                VAE_EPOCHS,
                MULTI_VAE,
                SEED,
            )
            train_dataset = gen.augment_dataset(train_dataset, AUGMENTATION, **augmentation_params)

        # train cnn
        results = train_model(
            model=CNNMNIST(),
            training_args=training_args,
            train_dataset=train_dataset,
            dev_dataset=val_dataset,
            test_dataset=test_dataset,
            seed=SEED,
        )
        # print the results
        print(results)