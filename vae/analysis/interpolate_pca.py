"""Data generation with convex hulls."""
import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import KernelPCA
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from utils import get_artifact_path, reshape_to_img
from vae.model_setup import load_model

# *** HYPERPARAMETERS ***

# vae parameters
VAE_EPOCHS = 50
VAE_Z_DIM = 2
VAE_ALPHA = 1.0
VAE_BETA = 1.0
# ignore these
VAE_N_EXAMPLES_LIMIT = None
VAE_TARGET_LABELS = None

N_SAMPLES = None
TARGET_CLASS = 9
KERNEL = "linear"


# *** Mlflow initialization ***

# initialize mlflow experiment & run
experiment = mlflow.get_experiment_by_name("Analysis")
if not experiment:
    experiment = mlflow.get_experiment(mlflow.create_experiment("Analysis"))
run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name="interpolate_pca")
artifact_path = get_artifact_path(run)

# log hyperparameters
mlflow.log_params(
    {
        "VAE_EPOCHS": VAE_EPOCHS,
        "VAE_Z_DIM": VAE_Z_DIM,
        "VAE_ALPHA": VAE_ALPHA,
        "VAE_BETA": VAE_BETA,
        "VAE_TARGET_LABELS": VAE_TARGET_LABELS,
        "VAE_N_EXAMPLES_LIMIT": VAE_N_EXAMPLES_LIMIT,
        "N_SAMPLES": N_SAMPLES,
        "TARGET_CLASS": TARGET_CLASS,
        "KERNEL": KERNEL,
    }
)


# *** Data preparation ***

try:
    (vae, device), _ = load_model(
        VAE_EPOCHS,
        VAE_Z_DIM,
        VAE_ALPHA,
        VAE_BETA,
        VAE_TARGET_LABELS,
        VAE_N_EXAMPLES_LIMIT,
        use_cuda=False,
    )
except LookupError:
    print("No Run with specified criteria found")
    exit(0)

# Load dataset
mnist = MNIST(
    root="~/torch_datasets",
    download=True,
    transform=ToTensor(),
    train=False,
)
dataset = mnist
# filter target labels
if VAE_TARGET_LABELS:
    indices = torch.where(
        torch.stack([(mnist.targets == t) for t in VAE_TARGET_LABELS]).sum(axis=0)
    )[0]
    dataset = torch.utils.data.Subset(mnist, indices)
dataloader = DataLoader(dataset, batch_size=512, shuffle=False)


# *** Analysis ***

# get means of encoded latent distributions
with torch.no_grad():
    latents, labels = zip(*[(vae.encoder(x.to(device))[0], y) for x, y in dataloader])
    latents = torch.cat(latents, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).numpy()

# filter target class
class_mask = labels == TARGET_CLASS
latents = latents[class_mask]
labels = labels[class_mask]

pca = KernelPCA(n_components=1, kernel=KERNEL, fit_inverse_transform=True).fit(latents)
pc = pca.transform(latents)

rows, cols = 8, 20
x = np.expand_dims(np.linspace(pc.min(), pc.max(), rows * cols), axis=1)
z = torch.Tensor(pca.inverse_transform(x))

with torch.no_grad():
    x_ = vae.decoder(z.to(device)).cpu().numpy()
    x_ = reshape_to_img(x_, 28, 28, rows, cols)
    plt.figure(figsize=(cols, rows))
    plt.imshow(x_, cmap="gray")
    plt.savefig(artifact_path / f"interpolation")
