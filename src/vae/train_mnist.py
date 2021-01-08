"""Training script for variational autoencoder on mnist."""
from pathlib import Path

from utils import get_artifact_path
from vae import vae_model

import mlflow
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

# *** Hyperparameters ***

EPOCHS = 20
Z_DIM = 2
ALPHA = 1.0
BETA = 1.0
TARGET_LABEL = 4


# *** Mlflow initialization ***

# initialize mlflow experiment & run
experiment = mlflow.get_experiment_by_name("VAE MNIST")
if not experiment:
    experiment = mlflow.get_experiment(mlflow.create_experiment("VAE MNIST"))
run = mlflow.start_run(
    experiment_id=experiment.experiment_id,
)
artifact_path = get_artifact_path(run)

# log hyperparameters
mlflow.log_params(
    {
        "epochs": EPOCHS,
        "z_dim": Z_DIM,
        "alpha": ALPHA,
        "beta": BETA,
        "target_label": TARGET_LABEL,
    }
)

# *** Model initialization ***

# Use cuda if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Model
vae = vae_model.VariationalAutoencoder(z_dim=Z_DIM).to(device)
# Optimizer
optim = torch.optim.Adam(vae.parameters())
# Loss
vae_loss = vae_model.vae_loss


# *** Data preparation ***

# Load datasets
mnist = MNIST(
    root="~/torch_datasets",
    download=True,
    transform=transforms.ToTensor(),
    train=True,
)
dataset = (
    torch.utils.data.Subset(mnist, torch.where(mnist.targets == TARGET_LABEL)[0])
    if TARGET_LABEL is not None
    else mnist
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# *** Training ***

vae.train()
step = 0
for epoch in range(EPOCHS):
    kld_losses, bce_losses = [], []
    with tqdm(total=len(dataloader)) as pbar:
        pbar.set_description(f"Train Epoch {epoch + 1}/{EPOCHS}", refresh=True)

        for x_true, _ in dataloader:
            step += 1
            x_true = x_true.to(device)

            # predict and compute loss
            x_hat, mean, variance_log = vae(x_true)
            bce_l, kld_l = vae_loss(x_hat, x_true, mean, variance_log, ALPHA, BETA)

            # update parameters
            optim.zero_grad()
            (bce_l + kld_l).backward()
            optim.step()

            # update losses
            mlflow.log_metrics(
                {
                    "binary_crossentropy_loss": bce_l.item(),
                    "kl_divergence_loss": kld_l.item(),
                },
                step=step,
            )

            # progress
            pbar.set_postfix({"bce-loss": bce_l.item(), "kld-loss": kld_l.item()})
            pbar.update(1)

mlflow.pytorch.save_model(
    vae,
    artifact_path / "model",
    code_paths=[Path(vae_model.__file__)],
)

mlflow.end_run()
