"""Trainer for VAEs."""
from typing import Callable, Dict, Optional, Tuple

import mlflow
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.models import BaseModel, ModelOutput

from .early_stopping import EarlyStopping
from .training_arguments import TrainingArguments


class Trainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: BaseModel,
        train_dataset: Dataset,
        dev_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        step_metrics: Optional[Callable] = None,
        epoch_metrics: Optional[Callable] = None,
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
        # Early Stopping
        if self.args.early_stopping:
            self.early_stop_monitor = EarlyStopping(n=self.args.early_stopping_window)

    def train(self) -> None:
        # seeding
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)

        # create dataloaders
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4
        )
        test_loader = DataLoader(self.dev_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)

        # raise error if both 'total steps' and 'epochs' provided in train args
        if self.args.total_steps is not None:
            self.total_steps = self.args.total_steps
            if self.args.epochs is not None:
                raise ValueError("Only specify 'epochs' or 'total_steps' in training arguments!")
        # use 'epochs' parameter if 'total_steps' is not defined
        elif self.args.epochs is not None:
            self.total_steps = self.args.epochs * len(train_loader)
        # raise error if none of 'epochs' or 'total_steps' is defined
        else:
            raise ValueError("You must either specify 'epochs' or 'total_steps' in training arguments!")
        # set validation intervall
        self.validation_intervall = self.args.validation_intervall or len(train_loader)

        # log config
        mlflow.log_params(self.args.__dict__)
        mlflow.log_params(self.model.config.__dict__)
        mlflow.log_param("train_dataset_size", len(self.train_dataset))
        mlflow.log_param("dev_dataset_size", len(self.dev_dataset))

        step_stopped, epoch_stopped = self.train_loop(train_loader, test_loader)

        # save final model
        if self.args.save_model and (self.args.save_intervall is None or step_stopped % self.args.save_intervall != 0):
            self.model.save(epoch_stopped)

    def train_loop(self, train_loader: DataLoader, test_loader: DataLoader) -> Tuple[int, int]:
        # initialize epoch & step counter
        epoch, step = 1, 0
        # gather outputs
        outputs = pd.DataFrame()
        # initialize train iterator
        train_iterator = iter(train_loader)
        # get the first batch
        batch = next(train_iterator)
        # set early stop initially to False
        early_stop = False
        with tqdm(total=self.total_steps, desc="Training epoch 1") as pbar:
            for step in range(1, self.total_steps + 1):
                # set model to train mode
                self.model.train()
                out = self.train_step(*[x.to(self.device, non_blocking=True) for x in batch])
                # train one step and remember outputs
                outputs = pd.concat(
                    [outputs, pd.DataFrame(out)],
                    ignore_index=True,
                )
                # compute step metrics
                metrics = self.log_step(outputs.iloc[-len(batch):], validate=False)
                # log to mlflow
                mlflow.log_metrics({"train_" + k: v for k, v in metrics.items()}, step=step)
                # progress bar
                pbar.set_postfix({k: f"{v:7.2f}" for k, v in metrics.items()})
                pbar.update(1)

                # validation on val dataset
                if step % self.validation_intervall == 0:
                    val_metrics, early_stop = self.validate(test_loader)
                    mlflow.log_metrics({"test_" + k: v for k, v in val_metrics.items()}, step=step)
                
                # optional save model
                if self.args.save_model and self.args.save_intervall and step % self.args.save_intervall == 0:
                    self.model.save(epoch)

                # get the next batch or None if end of trainloader is reached
                batch = next(train_iterator, None)
                # if end of trainloader is reached, it marks the end of an epoch
                if batch is None:
                    # re-initialize train_iterator
                    train_iterator = iter(train_loader)
                    # get the next batch
                    batch = next(train_iterator)
                    # compute epoch metrics
                    metrics = self.log_epoch(outputs, validate=False)
                    # log epoch metrics
                    mlflow.log_metrics({"train_" + k: v for k, v in metrics.items()}, step=epoch)
                    # reset outputs
                    outputs = pd.DataFrame()
                    # update epoch counter and tqdm description
                    if not step >= self.total_steps:
                        epoch += 1
                        pbar.set_description(f"Training epoch {epoch}", refresh=True)

                if early_stop:
                    break
                
        # return step, epoch stopped
        return step, epoch

    def validate(self, test_loader: DataLoader, evaluation: bool = False) -> Tuple[Dict[str, float], bool]:
        # set to eval mode
        self.model.eval()
        # gather outputs
        outputs = pd.DataFrame()
        # validate test_loader
        # with tqdm(total=len(test_loader), desc="Validation", leave=False) as test_pbar:
        for batch in test_loader:
            # test step, update running loss
            outputs = pd.concat(
                [outputs, pd.DataFrame(self.test_step(*[x.to(self.device, non_blocking=True) for x in batch]))],
                ignore_index=True,
            )
            metrics = self.log_step(outputs.iloc[-len(batch):], validate=not evaluation)
            # # progress bar
            # test_pbar.set_postfix({k: f"{v:7.2f}" for k, v in metrics.items()})
            # test_pbar.update(1)
        # compute epoch metrics and concat with step metrics of last step
        metrics = {**metrics, **self.log_epoch(outputs, validate=not evaluation)}
        # return validation metrics and early stop
        return metrics, self.args.early_stopping and self.early_stop_monitor(outputs["loss"].mean())

    def evaluate(self) -> Dict[str, float]:
        # create test_loader
        test_loader = DataLoader(self.test_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)
        # tmp save model
        model = self.model
        self.model = self.best_model
        # call validate
        print("Evaluating model.")
        metrics, _ = self.validate(test_loader)
        self.model = model
        # log to mlflow
        mlflow.log_metrics({"eval_" + k: v for k, v in metrics.items()})
        return {"eval_" + k: v for k, v in metrics.items()}

    def log_epoch(self, outputs: pd.DataFrame, validate: bool = False) -> Dict[str, float]:
        metrics = self.epoch_metrics(outputs["prediction"], outputs["label"], validate=validate) if self.epoch_metrics else {}
        # save best model
        if self.args.save_best_metric is not None and metrics.get(self.args.save_best_metric, False):
            self.best_model = self.model.__class__(self.model.config).to(self.model.config.device)
            self.best_model.load_state_dict(self.model.state_dict())
        return {
            "loss": outputs["loss"].mean(),
            **metrics,
        }

    def log_step(self, outputs: pd.DataFrame, validate: bool = False) -> Dict[str, float]:
        metrics = self.step_metrics(outputs["prediction"], outputs["label"], validate=validate) if self.step_metrics else {}
        # save best model
        if self.args.save_best_metric is not None and metrics.get(self.args.save_best_metric, False):
            self.best_model = self.model.__class__(self.model.config)
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
