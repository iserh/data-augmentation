"""Training script for variational autoencoder on mnist."""
import atexit
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
from tqdm import tqdm

import vae.celeba.model as model
from vae.celeba.model import VariationalAutoencoder, VAELoss
from utils import get_artifact_path

mlflow.set_tracking_uri("./experiments/CelebA")
finished = False


@atexit.register
def exit_handler():
    global finished
    if finished:
        mlflow.end_run()
    else:
        mlflow.end_run("KILLED")


class VAETrainer:
    def __init__(self, z_dim: int, beta: float) -> None:
        self.hparams = {"Z_DIM": z_dim, "BETA": beta}
        self.initialize_mlflow()

    def initialize_mlflow(self) -> None:
        # initialize mlflow experiment & run
        self.experiment = mlflow.get_experiment_by_name("VAE")
        if not self.experiment:
            self.experiment = mlflow.get_experiment(
                mlflow.create_experiment("VAE")
            )
        self.run = mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
        )
        self.artifact_path = get_artifact_path(self.run)

    def initialize_model(self, cuda: bool = True) -> None:
        # Use cuda if available
        self.device = "cuda:0" if cuda and torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # Model
        self.model = VariationalAutoencoder(z_dim=self.hparams["Z_DIM"]).to(
            self.device
        )
        # Optimizer
        self.optim = torch.optim.Adam(self.model.parameters())
        # Loss
        self.loss = VAELoss(beta=self.hparams["BETA"])

    def prepare_data(self) -> None:
        # Load dataset from torchvision
        celeba = CelebA(
            root="~/torch_datasets",
            transform=transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            split="train",
            target_type="identity",
            download=False,
        )
        print(f"Dataset size: {len(celeba)}")
        # create dataloader
        self.dataloader = DataLoader(
            celeba,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def train(self, epochs) -> None:
        self.hparams["EPOCHS"] = epochs
        mlflow.log_params(self.hparams)

        self.model.train()
        step = 0
        for e in range(epochs):
            with tqdm(total=len(self.dataloader)) as pbar:
                pbar.set_description(f"Train Epoch {e + 1}/{EPOCHS}", refresh=True)

                for x_true, _ in self.dataloader:
                    step += 1
                    x_true = x_true.to(self.device, non_blocking=True)

                    # predict and compute loss
                    output = self.model(x_true)
                    bce_l, kld_l = self.loss(x_true, *output)

                    # update parameters
                    self.optim.zero_grad()
                    (bce_l + kld_l).backward()
                    self.optim.step()

                    # update losses
                    mlflow.log_metrics(
                        {
                            "binary_crossentropy_loss": bce_l.item(),
                            "kl_divergence_loss": kld_l.item(),
                        },
                        step=step,
                    )

                    # progress
                    pbar.set_postfix(
                        {"bce-loss": bce_l.item(), "kld-loss": kld_l.item()}
                    )
                    pbar.update(1)

            if e % (epochs // 5) == 0:
                mlflow.pytorch.save_model(
                    self.model,
                    self.artifact_path / f"model-{e=}",
                    code_paths=[Path(model.__file__)],
                )


if __name__ == "__main__":

    EPOCHS = 250
    Z_DIM = 128
    BETA = 1.0

    vt = VAETrainer(z_dim=Z_DIM, beta=BETA)
    vt.initialize_model(cuda=True)
    vt.prepare_data()
    vt.train(epochs=EPOCHS)
    finished = True
