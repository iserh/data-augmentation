"""Training script for variational autoencoder on mnist."""

from typing import Optional, Tuple

import mlflow
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_artifact_path
from vae.model import VAELoss, VariationalAutoencoder

mlflow_run_success = False


class VAETrainer:
    """Trainer class for VAE training."""

    def __init__(self, z_dim: int, beta: float) -> None:
        """Initialize trainer. Set hyperparameters.

        Args:
            z_dim (int): Dimension of latent space
            beta (float): beta for KL-Divergence regularizer
        """
        self.hparams = {"Z_DIM": z_dim, "BETA": beta}
        self._initialize_mlflow()

    def initialize_model(self, model: VariationalAutoencoder, cuda: bool = True) -> None:
        # Use cuda if available
        self.device = "cuda:0" if cuda and torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # Model
        self.model = model.to(self.device)
        # Optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=5e-3)
        # Loss
        self.loss = VAELoss(beta=self.hparams["BETA"])

    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int, save_every: Optional[int]) -> None:
        # start new mlflow run
        self.run = mlflow.start_run(experiment_id=self.experiment.experiment_id, run_name="training")
        self.artifact_path = get_artifact_path(self.run)
        # log hyperparameters
        self.hparams["EPOCHS"] = epochs
        mlflow.log_params(self.hparams)

        for e in range(epochs):
            # Training
            self.model.train()
            running_bce_l, running_dkl_l = 0, 0
            with tqdm(total=len(train_loader)) as pbar:
                pbar.set_description(f"Train Epoch {e + 1}/{epochs}")
                for step, (x_true, _) in enumerate(train_loader, start=1):
                    # train step, compute losses, backward pass
                    bce_l, dkl_l = self._train_step(x_true.to(self.device, non_blocking=True))
                    running_bce_l += bce_l
                    running_dkl_l += dkl_l
                    # progress bar
                    pbar.set_postfix({"bce_l": running_bce_l / step, "dkl_l": running_dkl_l / step})
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
                pbar.set_description(f"Test Epoch {e + 1}/{epochs}")
                for step, (x_true, _) in enumerate(test_loader, start=1):
                    # train step, compute losses
                    bce_l, dkl_l = self._test_step(x_true.to(self.device, non_blocking=True))
                    running_bce_l += bce_l
                    running_dkl_l += dkl_l
                    # progress bar
                    pbar.set_postfix({"bce_l": running_bce_l / step, "dkl_l": running_dkl_l / step})
                    pbar.update(1)
            # log loss metrics
            mlflow.log_metrics(
                {"test_bce_l": running_bce_l / len(test_loader), "test_dkl_l": running_dkl_l / len(test_loader)}, step=e
            )
            # optional save model
            if save_every and e % save_every == 0 and e != epochs:
                mlflow.pytorch.save_model(
                    self.model,
                    self.artifact_path / f"model-epoch={e}",
                    code_paths=self.model.code_paths,
                )
        # save final model
        mlflow.pytorch.save_model(
            self.model,
            self.artifact_path / f"model-epoch={epochs}",
            code_paths=self.model.code_paths,
        )

    # *** private functions ***

    def _initialize_mlflow(self) -> None:
        # initialize mlflow experiment
        experiment_name = "VAE Training"
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            self.experiment = mlflow.get_experiment(mlflow.create_experiment(experiment_name))

    def _train_step(self, x_true: Tensor) -> Tuple[float, float]:
        # forward pass
        x_hat, mean, log_variance, _ = self.model(x_true)
        # compute losses
        bce_l, dkl_l = self.loss(x_true, x_hat, mean, log_variance)
        # update parameters
        self.optim.zero_grad()
        (bce_l + dkl_l).backward()
        self.optim.step()
        # return losses
        return bce_l.item(), dkl_l.item()

    @torch.no_grad()
    def _test_step(self, x_true: Tensor) -> Tuple[float, float]:
        # forward pass
        x_hat, mean, log_variance, _ = self.model(x_true)
        # compute losses
        bce_l, dkl_l = self.loss(x_true, x_hat, mean, log_variance)
        # return losses
        return bce_l.item(), dkl_l.item()


# exit handling
def exit_hook():
    global mlflow_run_success
    if mlflow_run_success:
        mlflow.end_run()
    else:
        mlflow.end_run("KILLED")


if __name__ == "__main__":
    import atexit

    from utils.config import mlflow_roots
    from vae.setup import get_dataloader, get_model

    atexit.register(exit_hook)

    DATASET = "MNIST"
    EPOCHS = 100
    Z_DIM = 2
    BETA = 1.0

    mlflow.set_tracking_uri(mlflow_roots[DATASET])
    train_loader = get_dataloader(DATASET, train=True)
    test_loader = get_dataloader(DATASET, train=False)
    model = get_model(DATASET, Z_DIM)

    vt = VAETrainer(Z_DIM, BETA)
    vt.initialize_model(model, cuda=True)
    vt.train(train_loader, test_loader, EPOCHS, save_every=20)

    mlflow_run_success = True
