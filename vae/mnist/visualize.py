"""Data generation with convex hulls."""
import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA, KernelPCA
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from utils import get_artifact_path
from vae.model_setup import load_model

# *** HYPERPARAMETERS ***

VAE_EPOCHS = 50
VAE_Z_DIM = 2
VAE_ALPHA = 1.0
VAE_BETA = 1.0
VAE_N_EXAMPLES_LIMIT = None
VAE_TARGET_LABELS = None

N_EXAMPLES = None
TRAIN = True
KERNEL = None
PROJECTION = None

kernel_limits = {
    None: [-4, 4],
    "linear": [-4, 4],
    "cosine": [-1.2, 1.2],
    "rbf": [-1, 1],
    "poly": [-4, 4],
    "sigmoid": [-1, 1],
}


# *** Mlflow initialization ***

# initialize mlflow experiment & run
experiment = mlflow.get_experiment_by_name("Feature Space")
if not experiment:
    experiment = mlflow.get_experiment(mlflow.create_experiment("Feature Space"))
run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name="encoded_space")
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
        "N_EXAMPLES": N_EXAMPLES,
        "TRAIN": TRAIN,
        "KERNEL": KERNEL,
        "PROJECTION": PROJECTION,
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
    mlflow.end_run("KILLED")
    print("No Run with specified criteria found")
    exit(0)

# Load dataset
mnist = MNIST(
    root="~/torch_datasets",
    download=True,
    transform=ToTensor(),
    train=TRAIN,
)
dataset = mnist
# filter target labels
if VAE_TARGET_LABELS:
    indices = torch.where(
        torch.stack([(mnist.targets == t) for t in VAE_TARGET_LABELS]).sum(axis=0)
    )[0]
    dataset = torch.utils.data.Subset(mnist, indices)
# set amount of examples to be visualized
if N_EXAMPLES:
    indices = torch.randperm(len(dataset))[:N_EXAMPLES]
    dataset = torch.utils.data.Subset(mnist, indices)
# create dataloader
dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

# *** Encoder Space Visualization ***

# get means of encoded latent distributions
with torch.no_grad():
    m_v_log, labels = zip(*[(vae.encoder(x.to(device)), y) for x, y in dataloader])
    means, variance_logs = zip(*m_v_log)
    means = torch.cat(means, dim=0).cpu().numpy()
    variance_logs = torch.cat(variance_logs, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).numpy()

unique_labels = np.unique(labels).astype("int")

means = (
    KernelPCA(n_components=2, kernel=KERNEL).fit(means).transform(means)
    if KERNEL
    else means
)


def draw_vector(v0, v1, ax=None, color=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle="->", linewidth=2, shrinkA=0, shrinkB=0, color=color)
    ax.annotate("", v1, v0, arrowprops=arrowprops)


# plot encoded latent means
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(
    xlim=kernel_limits[KERNEL], ylim=kernel_limits[KERNEL], projection=PROJECTION
)
for i in range(5, 6, 1):
    mask = labels == i
    points = means[mask]
    ax.scatter(*means[mask].T, s=1, color=get_cmap("tab10")(i), label=i, alpha=0.2)

    # principle component visualization
    pca = PCA(n_components=1).fit(points)
    # plot data
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v, color=get_cmap("tab10")(i))
    plt.axis("equal")

plt.legend()
plt.savefig(artifact_path / "generated_z.png")
plt.close()
