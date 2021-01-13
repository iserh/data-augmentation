"""Training script for variational autoencoder on mnist."""
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

import vae.vae_model as vae_model
from utils import get_artifact_path

# *** Hyperparameters ***

EPOCHS = 50
Z_DIM = 64
ALPHA = 1.0
BETA = 1.0
TARGET_LABELS = None
N_EXAMPLES_LIMIT = None


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
        "EPOCHS": EPOCHS,
        "Z_DIM": Z_DIM,
        "ALPHA": ALPHA,
        "BETA": BETA,
        "TARGET_LABELS": TARGET_LABELS,
        "N_EXAMPLES_LIMIT": N_EXAMPLES_LIMIT,
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

# Load dataset
mnist = MNIST(
    root="~/torch_datasets",
    download=True,
    transform=transforms.ToTensor(),
    train=True,
)
dataset = mnist
# filter target labels
if TARGET_LABELS:
    indices = torch.where(
        torch.stack([(mnist.targets == t) for t in TARGET_LABELS]).sum(axis=0)
    )[0]
    dataset = torch.utils.data.Subset(mnist, indices)
# limit number of examples used for training
if N_EXAMPLES_LIMIT:
    indices = torch.randperm(len(dataset))[:N_EXAMPLES_LIMIT]
    dataset = torch.utils.data.Subset(mnist, indices)
# create dataloader
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
