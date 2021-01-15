"""Data generation with convex hulls."""
import atexit
import torchvision.utils as vutils
import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms

from utils import get_artifact_path, TransformImage
from vae.celeba.trained_model import get_model


@atexit.register
def exit_handler():
    global finished
    if finished:
        mlflow.end_run()
    else:
        mlflow.end_run("KILLED")


finished = False


# *** HYPERPARAMETERS ***

VAE_EPOCHS = 20
VAE_Z_DIM = 100
VAE_BETA = 1.0

N_SAMPLES = 512


# *** Mlflow initialization ***

# initialize mlflow experiment & run
mlflow.set_tracking_uri("experiments/CelebA")
experiment = mlflow.get_experiment_by_name("Generation")
if not experiment:
    experiment = mlflow.get_experiment(mlflow.create_experiment("Generation"))
run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name="random")
artifact_path = get_artifact_path(run)
(artifact_path / "data").mkdir(exist_ok=True, parents=True)

# log hyperparameters
mlflow.log_params(
    {
        "VAE_EPOCHS": VAE_EPOCHS,
        "VAE_Z_DIM": VAE_Z_DIM,
        "VAE_BETA": VAE_BETA,
        "N_SAMPLES": N_SAMPLES,
    }
)


# *** Data preparation ***

try:
    (vae, device), _ = get_model(
        VAE_EPOCHS,
        VAE_Z_DIM,
        VAE_BETA,
    )
except LookupError:
    print("No Run with specified criteria found")
    exit()

to_image_trans = TransformImage(channels=3, height=64, width=64, mode="RGB")


# *** Generation ***

print("Sampling from datasets")
# extend / shrink dataset to N_SAMPLES
z = torch.empty(size=(N_SAMPLES, VAE_Z_DIM)).normal_(0, 1)

# generate images
with torch.no_grad():
    x_ = vae.decoder(z.to(device))

# Plot the real images
plt.figure(figsize=(15,15))
# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(x_[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig(artifact_path / "fake_real.png")

finished = True
