from typing import Dict

import mlflow
import torch
from numpy import ceil
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import TensorDataset

from utils.data import load_datasets
from utils.mlflow import backend_stores
from utils.models.model_config import ModelConfig
from utils.trainer import Trainer, TrainingArguments

import generative_classifier
from generative_classifier.models.architectures import GenerativeClassifierV1

generative_classifier.models.base.model_store = "generative_classifiers/MNIST"


def add_noise_to_dataset(dataset: Dataset, std: float = 0.3) -> TensorDataset:
    X, Y = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    uniform = torch.rand(size=(X.size(0) // 2, *X.size()[1:]))
    mean = X.mean(dim=0, keepdim=True)
    idx = torch.randperm(X.size(0))[: X.size(0) // 2]
    normal: Tensor = torch.cat([torch.normal(mean, std) for _ in range(X.size(0) // 2)], dim=0)
    normal += X[idx]
    _min = normal.flatten(-2).min(-1)[0].unsqueeze(-1).unsqueeze(-1)
    _max = normal.flatten(-2).max(-1)[0].unsqueeze(-1).unsqueeze(-1)
    normal = (normal - _min) / (_max - _min)
    X = torch.cat([X, uniform, normal])
    Y = torch.cat(
        [torch.ones(size=(Y.size(0),), dtype=torch.long), *[torch.zeros(size=(Y.size(0) // 2,), dtype=torch.long)] * 2],
        dim=0,
    )
    return TensorDataset(X, Y)


def metrics(predictions: Tensor, labels: Tensor, validate: bool) -> Dict[str, float]:
    acc = accuracy_score(labels, predictions)
    return {"acc": acc}


DATASET = "MNIST"
_, train_dataset, val_dataset, _ = load_datasets(DATASET)

train_dataset = add_noise_to_dataset(train_dataset)
val_dataset = add_noise_to_dataset(val_dataset)

training_args = TrainingArguments(
    epochs=1, seed=1337, batch_size=512, validation_intervall=ceil(len(train_dataset) / 512), no_cuda=True
)

trainer = Trainer(
    args=training_args,
    model=GenerativeClassifierV1(ModelConfig()),
    train_dataset=train_dataset,
    dev_dataset=val_dataset,
    epoch_metrics=metrics,
)

mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
# start mlflow run in experiment
mlflow.set_experiment(f"Generative Classifier")

with mlflow.start_run():
    trainer.train()
