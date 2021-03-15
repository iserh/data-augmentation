"""Trainer for VAEs."""
from typing import Callable, Dict, Optional

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from numpy import ceil

from utils.mlflow import mlflow_active, mlflow_available
from utils.models import BaseModel, ModelOutput
from torch.optim import Adam

from .training_arguments import TrainingArguments

if mlflow_available:
    import mlflow


class Trainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: BaseModel,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None,
    ) -> None:
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        # Use cuda if available
        self.device = "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        # Model
        self.model = model.to(self.device)
        # Optimizer
        self.optim = Adam(self.model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    def train(self) -> None:
        # seeding
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)

        # create dataloaders
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4
        )
        test_loader = (
            DataLoader(self.test_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)
            if self.test_dataset is not None
            else None
        )

        # log config
        if mlflow_active():
            mlflow.log_params(self.args.__dict__)
            mlflow.log_param("total_steps", self.args.epochs * len(train_loader))
            mlflow.log_params(self.model.config.__dict__)
            mlflow.log_param("train_dataset_size", len(self.train_dataset))
            if self.test_dataset is not None:
                mlflow.log_param("test_dataset_size", len(self.test_dataset))

        # reset step counter
        self.step = 0
        # reset best model tracker
        if self.args.metric_for_best_model is not None:
            self.best_score = 0
        # start training
        with tqdm(total=self.args.epochs * len(train_loader)) as pbar:
            for e in range(1, self.args.epochs + 1):
                # update tqdm description
                pbar.set_description(f"Training epoch {e}/{self.args.epochs}", refresh=True)
                # train epoch
                self.train_epoch(e, train_loader, pbar)
                # test epoch
                if test_loader is not None:
                    self.test_epoch(test_loader, e, pbar)
                # optional save model
                if self.args.save_model and self.args.save_epochs and e % self.args.save_epochs == 0:
                    self.model.save(e)

        # save final model
        if self.args.save_model:
            self.model.save(self.args.epochs)

    def train_epoch(self, epoch: int, train_loader: DataLoader, pbar: tqdm) -> None:
        # output df for metric calculation
        outputs = pd.DataFrame()
        # set model to train mode
        self.model.train()
        for batch in train_loader:
            # make a train step, collect output
            out = self.train_step(*[t.to(self.device, non_blocking=True) for t in batch])
            # append output to outputs df
            outputs = pd.concat(
                [outputs, pd.DataFrame(out)],
                ignore_index=True,
            )
            # compute step metrics
            metrics = self._compute_metrics(outputs.iloc[-len(batch) :], update_best_metric=False)
            # log to mlflow
            if (
                mlflow_active()
                and self.args.log_steps is not None
                and self.step % self.args.log_steps == 0
                and self.step != 0
            ):
                mlflow.log_metrics({"train_" + k: v for k, v in metrics.items()}, step=self.step)
            # progress bar
            pbar.set_postfix({k: f"{v:7.2f}" for k, v in metrics.items()})
            pbar.update()

            self.step += 1

        # compute epoch metrics
        metrics = self._compute_metrics(outputs, update_best_metric=False)
        # log epoch metrics
        if mlflow_active():
            mlflow.log_metrics({f"train_{k}_epoch": v for k, v in metrics.items()}, step=epoch)

    def test_epoch(self, test_loader: DataLoader, epoch: Optional[int] = None, pbar: Optional[tqdm] = None) -> None:
        # output df for metric calculation
        outputs = pd.DataFrame()
        # set model to train mode
        self.model.eval()
        for batch in test_loader:
            # make a train step, collect output
            out = self.test_step(*[t.to(self.device, non_blocking=True) for t in batch])
            # append output to outputs df
            outputs = pd.concat(
                [outputs, pd.DataFrame(out)],
                ignore_index=True,
            )

        # compute epoch metrics
        metrics = self._compute_metrics(outputs, True)
        # log epoch metrics
        if mlflow_active():
            mlflow.log_metrics({f"test_{k}_epoch" if k.find("best") == -1 else k: v for k, v in metrics.items()}, step=epoch)

    def evaluate(self) -> Dict[str, float]:
        # create test_loader
        test_loader = DataLoader(self.test_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)
        # output df for metric calculation
        outputs = pd.DataFrame()
        tmp_model = self.model
        self.model = self.best_model
        # set model to train mode
        self.model.eval()
        for batch in test_loader:
            # make a train step, collect output
            out = self.test_step(*[t.to(self.device, non_blocking=True) for t in batch])
            # append output to outputs df
            outputs = pd.concat(
                [outputs, pd.DataFrame(out)],
                ignore_index=True,
            )
        self.model = tmp_model

        # compute epoch metrics
        metrics = self._compute_metrics(outputs, False)
        # log epoch metrics
        if mlflow_active():
            mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
        return metrics

    def _compute_metrics(self, outputs: pd.DataFrame, update_best_metric: bool = False) -> Dict[str, float]:
        metrics = self.compute_metrics(outputs["prediction"], outputs["label"]) if self.compute_metrics else {}
        # save best model
        if (
            update_best_metric
            and self.args.metric_for_best_model is not None
            and metrics.get(self.args.metric_for_best_model, False)
            and metrics[self.args.metric_for_best_model] > self.best_score
        ):
            self.best_score = metrics[self.args.metric_for_best_model]
            metrics[f"best_{self.args.metric_for_best_model}"] = self.best_score
            self.best_model = self.model.__class__(self.model.config).to(self.model.config.device)
            self.best_model.load_state_dict(self.model.state_dict())
        return {
            "loss": outputs["loss"].mean(),
            **metrics,
        }

    def train_step(self, x: Tensor, y: Optional[Tensor] = None) -> Dict[str, float]:
        # forward pass
        output: ModelOutput = self.model(x, y)
        # update parameters
        self.optim.zero_grad()
        output.loss.backward()
        self.optim.step()
        # return losses
        return {"loss": output.loss.item(), "prediction": output.prediction.detach().cpu(), "label": y.cpu()}

    @torch.no_grad()
    def test_step(self, x: Tensor, y: Optional[Tensor] = None) -> Dict[str, float]:
        # forward pass
        output: ModelOutput = self.model(x, y)
        # return losses
        return {"loss": output.loss.item(), "prediction": output.prediction.detach().cpu(), "label": y.cpu()}
