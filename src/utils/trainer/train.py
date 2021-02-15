"""Trainer for VAEs."""
from typing import Callable, Dict, Optional

import mlflow
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.models import BaseModel, ModelOutput

from .training_arguments import TrainingArguments


class Trainer:
    def __init__(
        self, args: TrainingArguments, model: BaseModel, train_dataset: Dataset, dev_dataset: Dataset, test_dataset: Optional[Dataset] = None, step_metrics: Optional[Callable] = None, epoch_metrics: Optional[Callable] = None
    ) -> None:
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.step_metrics = step_metrics
        self.epoch_metrics = epoch_metrics
        # Use cuda if available
        self.device = "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        # Model
        self.model = model.to(self.device)
        # Optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=5e-3)

    def train(self) -> None:
        # log config
        mlflow.log_params(self.args.__dict__)
        mlflow.log_params(self.model.config.__dict__)
        mlflow.log_param("train_dataset_size", len(self.train_dataset))
        mlflow.log_param("dev_dataset_size", len(self.dev_dataset))
        # seeding
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
        # create dataloaders
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
        test_loader = DataLoader(self.dev_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)

        with tqdm(total=self.args.epochs * len(train_loader)) as pbar:
            for epoch in range(1, self.args.epochs + 1, 1):
                # update description
                pbar.set_description(f"Training Epoch {epoch}/{self.args.epochs}", refresh=True)
                # train for one epoch
                epoch_metrics = self.train_epoch(epoch, train_loader, pbar)
                # log loss metrics
                mlflow.log_metrics(epoch_metrics)
                # test for one epoch
                epoch_metrics = self.test_epoch(epoch, test_loader)
                # log loss metrics
                mlflow.log_metrics(epoch_metrics)
                # optional save model
                if self.args.save_model and self.args.save_intervall and epoch % self.args.save_intervall == 0 and epoch != self.args.epochs:
                    self.model.save(epoch)
        # save final model
        if self.args.save_model:
            self.model.save(self.args.epochs)
    
    def evaluate(self) -> Dict[str, float]:
        mlflow.log_param("test_dataset_size", len(self.test_dataset))
        # create dataloader
        test_loader = DataLoader(self.test_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)
        # evaluate whole test_loader once
        self.model.eval()
        running_loss = 0
        predictions, labels = torch.Tensor([]), torch.Tensor([])
        with tqdm(total=len(test_loader), desc="Evaluating") as pbar:
            for step, (x, y) in enumerate(test_loader, start=1):
                # test step, update running loss
                output = self.test_step(x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))
                running_loss += output["loss"]
                predictions = torch.cat([predictions, output["predictions"]], dim=0)
                labels = torch.cat([labels, y.flatten()], dim=0)
                metrics = self.step_metrics(predictions, labels) if self.step_metrics else {}
                # progress bar
                pbar.set_postfix({
                    "loss": running_loss / step,
                    **metrics,
                })
                pbar.update(1)
        epoch_metrics = self.epoch_metrics(predictions, labels) if self.epoch_metrics else {}
        metrics = {
            "eval_loss": running_loss / len(test_loader),
            **{"eval_" + k: v for k, v in epoch_metrics.items()},
            **{"eval_" + k: v for k, v in metrics.items()},
        }
        # log loss metrics
        mlflow.log_metrics(metrics)
        return metrics

    def train_epoch(self, epoch: int, train_loader: DataLoader, pbar: tqdm) -> Dict[str, float]:
        self.model.train()
        running_loss = 0
        predictions, labels = torch.Tensor([]), torch.Tensor([])
        for step, (x, y) in enumerate(train_loader, start=1):
            # train step, update running loss
            output = self.train_step(x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))
            running_loss += output["loss"]
            predictions = torch.cat([predictions, output["predictions"]], dim=0)
            labels = torch.cat([labels, y.flatten()], dim=0)
            metrics = self.step_metrics(predictions, labels) if self.step_metrics else {}
            # log to mlflow
            mlflow.log_metrics({
                "epoch": epoch - 1 + (0.5 * step / len(train_loader)),
                **{"train_" + k: v for k, v in metrics.items()},
            })
            # progress bar
            pbar.set_postfix({
                "loss": running_loss / step,
                **metrics
            })
            pbar.update(1)
        epoch_metrics = self.epoch_metrics(predictions, labels) if self.epoch_metrics else {}
        # return runnning loss
        return {"train_loss": running_loss / len(train_loader), **{"train_" + k: v for k, v in epoch_metrics.items()}}

    def test_epoch(self, epoch: int, test_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0
        predictions, labels = torch.Tensor([]), torch.Tensor([])
        with tqdm(total=len(test_loader), desc=f"Test epoch {epoch}/{self.args.epochs}", leave=True) as test_pbar:
            for step, (x, y) in enumerate(test_loader, start=1):
                # test step, update running loss
                output = self.test_step(x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))
                running_loss += output["loss"]
                predictions = torch.cat([predictions, output["predictions"]], dim=0)
                labels = torch.cat([labels, y.flatten()], dim=0)
                metrics = self.step_metrics(predictions, labels) if self.step_metrics else {}
                # log to mlflow
                mlflow.log_metrics({
                    "epoch": epoch - 0.5 + (0.5 * step / len(test_loader)),
                    **{"test_" + k: v for k, v in metrics.items()}
                })
                # progress bar
                test_pbar.set_postfix({
                    "loss": running_loss / step,
                    **metrics,
                })
                test_pbar.update(1)
        epoch_metrics = self.epoch_metrics(predictions, labels) if self.epoch_metrics else {}
        # return runnning loss
        return {"test_loss": running_loss / len(test_loader), **{"test_" + k: v for k, v in epoch_metrics.items()}}

    def train_step(self, x: Tensor, y: Optional[Tensor] = None) -> Dict[str, float]:
        # forward pass
        output: ModelOutput = self.model(x, y)
        # update parameters
        self.optim.zero_grad()
        output.loss.backward()
        self.optim.step()
        # return losses
        return {"loss": output.loss.item(), "predictions": output.prediction.cpu()}

    @torch.no_grad()
    def test_step(self, x: Tensor, y: Optional[Tensor] = None) -> Dict[str, float]:
        # forward pass
        output: ModelOutput = self.model(x, y)
        # return losses
        return {"loss": output.loss.item(), "predictions": output.prediction.cpu()}
