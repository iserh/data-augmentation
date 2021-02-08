"""Trainer for CNNs."""
from typing import Dict, Optional
from utils import uri_to_path
from torch.utils.data import DataLoader, Dataset

import mlflow
import torch
from torch import Tensor
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from dataclasses import dataclass


@dataclass
class TrainingArguments:
    epochs: int
    save_intervall: Optional[int] = None
    save_model: bool = True


class Trainer:

    def __init__(self, args: TrainingArguments, model: nn.Module, train_dataset: Dataset, test_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> None:
        # copy training arguments
        self.args = args
        # train dataset, test dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_dataset = eval_dataset

        # Use cuda if available
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Model
        self.model = model.to(self.device)
        # Optimizer
        self.optim = torch.optim.Adam(self.model.parameters())
        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def train(self) -> None:
        # set output_dir to artifact path
        output_dir = uri_to_path(mlflow.active_run().info.artifact_uri)
        # log config
        mlflow.log_params(self.args.__dict__)
        mlflow.log_params({"train_dataset_size": len(self.train_dataset), "test_dataset_size": len(self.test_dataset)})
        # create dataloaders
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
        test_loader = DataLoader(self.test_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)

        with tqdm(total=self.args.epochs * len(train_loader)) as pbar:
            for e in range(self.args.epochs):
                pbar.set_description(f"Training Epoch {e}/{self.args.epochs}", refresh=True)

                # Training
                self.model.train()
                running_loss = 0
                predictions, targets = torch.LongTensor([]), torch.LongTensor([])
                for step, (x, y) in enumerate(train_loader, start=1):
                    # train step, compute losses, backward pass
                    loss, pred = self._train_step(x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))
                    running_loss += loss
                    predictions = torch.cat([predictions, pred], dim=0)
                    targets = torch.cat([targets, y.flatten()], dim=0)
                    # progress bar
                    pbar.set_postfix({
                        "loss": running_loss / step,
                        "acc": accuracy_score(targets, predictions),
                        "f1": f1_score(targets, predictions, average="weighted"),
                    })
                    pbar.update(1)
                # log loss metrics
                mlflow.log_metrics(
                    {
                        "train_loss": running_loss / len(train_loader),
                        "train_acc": accuracy_score(targets, predictions),
                        "train_f1": f1_score(targets, predictions, average="weighted"),
                    },
                    step=e,
                )

                # Evaluation
                self.model.eval()
                running_loss = 0
                predictions, targets = torch.LongTensor([]), torch.LongTensor([])
                with tqdm(total=len(test_loader), desc=f"Dev test epoch {e + 1}/{self.args.epochs}", leave=False) as eval_pbar:
                    for step, (x, y) in enumerate(test_loader, start=1):
                        # train step, compute losses, backward pass
                        loss, pred = self._test_step(x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))
                        running_loss += loss
                        predictions = torch.cat([predictions, pred], dim=0)
                        targets = torch.cat([targets, y.flatten()], dim=0)
                        # progress bar
                        eval_pbar.set_postfix({
                            "loss": running_loss / step,
                            "acc": accuracy_score(targets, predictions),
                            "f1": f1_score(targets, predictions, average="weighted"),
                        })
                        eval_pbar.update(1)
                # log loss metrics
                mlflow.log_metrics(
                    {
                        "test_loss": running_loss / len(test_loader),
                        "test_acc": accuracy_score(targets, predictions),
                        "test_f1": f1_score(targets, predictions, average="weighted"),
                    },
                    step=e,
                )
                # optional save model
                if self.args.save_model and self.args.save_intervall and e % self.args.save_intervall == 0 and e != self.args.epochs:
                    mlflow.pytorch.save_model(
                        self.model,
                        output_dir / f"models/model-epoch={e}",
                        code_paths=self.model.code_paths,
                    )
            # save final model
            if self.args.save_model:
                mlflow.pytorch.save_model(
                    self.model,
                    output_dir / f"models/model-epoch={self.args.epochs}",
                    code_paths=self.model.code_paths,
                )
    
    def evaluate(self) -> Dict[str, float]:
        mlflow.log_params({"eval_dataset_size": len(self.eval_dataset)})
        # create dataloaders
        eval_loader = DataLoader(self.eval_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)

        # Evaluation
        self.model.eval()
        running_loss = 0
        predictions, targets = torch.LongTensor([]), torch.LongTensor([])
        with tqdm(total=len(eval_loader), desc="Evaluating model") as pbar:
            for step, (x, y) in enumerate(eval_loader, start=1):
                # train step, compute losses, backward pass
                loss, pred = self._test_step(x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))
                running_loss += loss
                predictions = torch.cat([predictions, pred], dim=0)
                targets = torch.cat([targets, y.flatten()], dim=0)
                # progress bar
                pbar.set_postfix({
                    "loss": running_loss / step,
                    "acc": accuracy_score(targets, predictions),
                    "f1": f1_score(targets, predictions, average="weighted"),
                })
                pbar.update(1)
        # compute scores
        scores = {
            "eval_loss": running_loss / len(eval_loader),
            "eval_acc": accuracy_score(targets, predictions),
            "eval_f1": f1_score(targets, predictions, average="weighted"),
        }
        # log loss metrics
        mlflow.log_metrics(scores)
        return scores

    # *** private functions ***

    def _train_step(self, x: Tensor, y_true: Tensor) -> float:
        # forward pass
        y_pred = self.model(x)
        # compute losses
        loss = self.criterion(y_pred, y_true.flatten())
        # update parameters
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        # return losses
        return loss.item(), y_pred.argmax(dim=-1).cpu()

    @torch.no_grad()
    def _test_step(self, x: Tensor, y_true: Tensor) -> float:
        # forward pass
        y_pred = self.model(x)
        # compute losses
        loss = self.criterion(y_pred, y_true.flatten())
        # return losses
        return loss.item(), y_pred.argmax(dim=-1).cpu()
