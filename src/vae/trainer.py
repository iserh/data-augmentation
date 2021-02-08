"""Trainer for VAEs."""
from dataclasses import dataclass
from typing import Optional, Tuple, Type
from vae.models.base import VAEConfig

import mlflow
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import uri_to_path
from utils.data import Datasets
from utils.integrations import BackendStore, ExperimentName
from vae.models import VAEModel, VAEOutput


@dataclass
class VAETrainingArguments:
    save_intervall: Optional[int] = None
    no_cuda: bool = False


class VAETrainer:
    def __init__(
        self, args: VAETrainingArguments, model: VAEModel, train_dataset: Dataset, eval_dataset: Dataset
    ) -> None:
        self.args = args or VAETrainingArguments()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        # Use cuda if available
        self.device = "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"

        # Model
        self.model = model.to(self.device)
        self.model.config.compute_loss = True
        # Optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=5e-3)

    def train(self) -> None:
        # set output dir to the artifact path of the active run
        output_dir = uri_to_path(mlflow.active_run().info.artifact_uri)
        # log config
        mlflow.log_params(self.args.__dict__)
        mlflow.log_param("z_dim", self.model.config.z_dim)
        # create dataloaders
        train_loader = DataLoader(self.train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
        test_loader = DataLoader(self.eval_dataset, batch_size=512, shuffle=False, pin_memory=True, num_workers=4)

        with tqdm(total=self.model.config.epochs * len(train_loader)) as pbar:
            for e in range(1, self.model.config.epochs, 1):
                pbar.set_description(f"Training Epoch {e}/{self.model.config.epochs}", refresh=True)

                # Training
                self.model.train()
                running_bce_l, running_dkl_l = 0, 0
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
                with tqdm(total=len(test_loader), desc=f"Dev test epoch {e}/{self.model.config.epochs}", leave=False) as eval_pbar:
                    for step, (x_true, _) in enumerate(test_loader, start=1):
                        # train step, compute losses
                        bce_l, dkl_l = self._test_step(x_true.to(self.device, non_blocking=True))
                        running_bce_l += bce_l
                        running_dkl_l += dkl_l
                        # progress bar
                        eval_pbar.set_postfix({"bce_l": running_bce_l / step, "dkl_l": running_dkl_l / step})
                        mlflow.log_metric("epoch", e + 0.5 + 0.5 * step / len(test_loader))
                        eval_pbar.update(1)

                # log loss metrics
                mlflow.log_metrics(
                    {"test_bce_l": running_bce_l / len(test_loader), "test_dkl_l": running_dkl_l / len(test_loader)}, step=e
                )
                # optional save model
                if self.args.save_intervall and e % self.args.save_intervall == 0 and e != self.args.epochs:
                    mlflow.pytorch.save_model(
                        self.model,
                        output_dir / f"models/model-epoch={e}",
                        code_paths=self.model.code_paths,
                    )

        # save final model
        mlflow.pytorch.save_model(
            self.model,
            output_dir / f"models/model-epoch={self.model.config.epochs}",
            code_paths=self.model.code_paths,
        )

    # *** private functions ***

    def _train_step(self, x_true: Tensor) -> Tuple[float, float]:
        # forward pass
        output: VAEOutput = self.model(x_true)
        bce_l, dkl_l = output.loss.reconstruction, output.loss.kl_divergence
        # update parameters
        self.optim.zero_grad()
        (bce_l + dkl_l).backward()
        self.optim.step()
        # return losses
        return bce_l.item(), dkl_l.item()

    @torch.no_grad()
    def _test_step(self, x_true: Tensor) -> Tuple[float, float]:
        # forward pass
        output: VAEOutput = self.model(x_true)
        # return losses
        return output.loss.reconstruction.item(), output.loss.kl_divergence.item()


def train(model_class: Type[VAEModel], vae_config: VAEConfig, dataset: str, args: Optional[VAETrainingArguments] = None) -> None:
    # initialize mlflow experiment
    mlflow.set_tracking_uri(BackendStore[dataset].value)
    mlflow.set_experiment(ExperimentName.VAETrain.value)

    # load train/test data
    train_dataset = Datasets(dataset, train=True)
    eval_dataset = Datasets(dataset, train=False)
    # create vae model
    model = model_class(vae_config)
    # initialize trainer
    trainer = VAETrainer(
        args=args or VAETrainingArguments(),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    # train model
    with mlflow.start_run():
        trainer.train()


if __name__ == "__main__":
    from vae.models.model_v1 import VAEModelV1
    DATASET = "MNIST"
    CONFIG = VAEConfig(epochs=50, z_dim=50, beta=1.0)
    train(VAEModelV1, CONFIG, DATASET, VAETrainingArguments(save_intervall=50))
    CONFIG = VAEConfig(epochs=50, z_dim=100, beta=1.0)
    train(VAEModelV1, CONFIG, DATASET, VAETrainingArguments(save_intervall=50))
