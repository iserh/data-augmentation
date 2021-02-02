"""Trainer for VAEs."""
from dataclasses import dataclass
from typing import Optional, Tuple

import mlflow
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import uri_to_path
from utils.data import Datasets
from utils.integrations import BackendStore, ExperimentName
from vae.loss import VAELoss
from vae.models import MNISTVAE, VAEBaseModel


@dataclass
class VAETrainingArguments:
    epochs: int
    beta: float
    save_intervall: Optional[int] = None
    no_cuda: bool = False


class VAETrainer:
    def __init__(
        self, args: VAETrainingArguments, model: VAEBaseModel, train_dataset: Dataset, eval_dataset: Dataset
    ) -> None:
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        # Use cuda if available
        self.device = "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        print("Using device:", self.device)

        # Model
        self.model = model.to(self.device)
        # Optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=5e-3)
        # Loss
        self.criterion = VAELoss(beta=args.beta)

    def train(self) -> None:
        # set output dir to the artifact path of the active run
        output_dir = uri_to_path(mlflow.active_run().info.artifact_uri)
        # log config
        mlflow.log_params(self.args.__dict__)
        mlflow.log_param("z_dim", self.model.z_dim)
        # create dataloaders
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
        test_loader = DataLoader(self.eval_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)

        for e in range(self.args.epochs):
            # Training
            self.model.train()
            running_bce_l, running_dkl_l = 0, 0
            with tqdm(total=len(train_loader)) as pbar:
                pbar.set_description(f"Train Epoch {e + 1}/{self.args.epochs}")
                for step, (x_true, _) in enumerate(train_loader, start=1):
                    # train step, compute losses, backward pass
                    bce_l, dkl_l = self._train_step(x_true.to(self.device, non_blocking=True))
                    running_bce_l += bce_l
                    running_dkl_l += dkl_l
                    # progress bar
                    pbar.set_postfix({"bce_l": running_bce_l / step, "dkl_l": running_dkl_l / step})
                    mlflow.log_metric("epoch", e + 0.5 * step / len(train_loader))
                    pbar.update(1)
            # log loss metrics
            mlflow.log_metrics(
                {"train_bce_l": running_bce_l / len(train_loader), "train_dkl_l": running_dkl_l / len(train_loader)},
                step=e,
            )

            # Evaluation
            self.model.eval()
            running_bce_l, running_dkl_l = 0, 0
            with tqdm(total=len(test_loader)) as pbar:
                pbar.set_description(f"Test Epoch {e + 1}/{self.args.epochs}")
                for step, (x_true, _) in enumerate(test_loader, start=1):
                    # train step, compute losses
                    bce_l, dkl_l = self._test_step(x_true.to(self.device, non_blocking=True))
                    running_bce_l += bce_l
                    running_dkl_l += dkl_l
                    # progress bar
                    pbar.set_postfix({"bce_l": running_bce_l / step, "dkl_l": running_dkl_l / step})
                    mlflow.log_metric("epoch", e + 0.5 + 0.5 * step / len(test_loader))
                    pbar.update(1)
            # log loss metrics
            mlflow.log_metrics(
                {"test_bce_l": running_bce_l / len(test_loader), "test_dkl_l": running_dkl_l / len(test_loader)}, step=e
            )
            # optional save model
            if self.args.save_intervall and e % self.args.save_intervall == 0 and e != self.args.epochs and e > 0:
                mlflow.pytorch.save_model(
                    self.model,
                    output_dir / f"models/model-epoch={e}",
                    code_paths=self.model.code_paths,
                )
        # save final model
        mlflow.pytorch.save_model(
            self.model,
            output_dir / f"models/model-epoch={self.args.epochs}",
            code_paths=self.model.code_paths,
        )

    # *** private functions ***

    def _train_step(self, x_true: Tensor) -> Tuple[float, float]:
        # forward pass
        x_hat, mean, log_variance = self.model(x_true)
        # compute losses
        bce_l, dkl_l = self.criterion(x_hat, x_true, mean, log_variance)
        # update parameters
        self.optim.zero_grad()
        (bce_l + dkl_l).backward()
        self.optim.step()
        # return losses
        return bce_l.item(), dkl_l.item()

    @torch.no_grad()
    def _test_step(self, x_true: Tensor) -> Tuple[float, float]:
        # forward pass
        x_hat, mean, log_variance = self.model(x_true)
        # compute losses
        bce_l, dkl_l = self.criterion(x_hat, x_true, mean, log_variance)
        # return losses
        return bce_l.item(), dkl_l.item()


def train(args: VAETrainingArguments, z_dim: int, dataset: str) -> None:
    # initialize mlflow experiment
    mlflow.set_tracking_uri(BackendStore[dataset].value)
    mlflow.set_experiment(ExperimentName.VAETrain.value)

    # load train/test data
    train_dataset = DATASETS[dataset](train=True)
    eval_dataset = DATASETS[dataset](train=False)
    # create vae model
    model = MNISTVAE(z_dim)
    # initialize trainer
    trainer = VAETrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    # train model
    with mlflow.start_run():
        trainer.train()


if __name__ == "__main__":
    DATASET = "MNIST"

    for z_dim in [10, 20, 50, 100]:
        training_args = VAETrainingArguments(epochs=100, beta=1.0, save_intervall=20)
        train(training_args, z_dim, DATASET)
