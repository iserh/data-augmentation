"""Data generation with convex hulls."""
import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import get_cmap
from numpy.linalg import det
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import dirichlet
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from utils import get_artifact_path, reshape_to_img
from vae.mnist.model_setup import load_model


def dist_in_hull(points: np.ndarray, n: int) -> np.ndarray:
    """Function stolen from stackoverflow."""
    dims = points.shape[-1]
    hull = points[ConvexHull(points).vertices]
    deln = points[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)
    sample = np.random.choice(len(vols), size=n, p=vols / vols.sum())

    return np.einsum(
        "ijk, ij -> ik", deln[sample], dirichlet.rvs([1] * (dims + 1), size=n)
    )


# *** HYPERPARAMETERS ***

VAE_EPOCHS = 50
VAE_Z_DIM = 2
VAE_ALPHA = 1.0
VAE_BETA = 1.0
VAE_N_EXAMPLES_LIMIT = None
VAE_TARGET_LABELS = None

N_EXAMPLES = None
N_SAMPLES = 10_000
OUTLIER_CONTAMINATION = 0.25


# *** Mlflow initialization ***

# initialize mlflow experiment & run
experiment = mlflow.get_experiment_by_name("MNIST Generation")
if not experiment:
    experiment = mlflow.get_experiment(mlflow.create_experiment("MNIST Generation"))
run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name="convex_hull")
artifact_path = get_artifact_path(run)
(artifact_path / "data").mkdir(exist_ok=True, parents=True)

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
        "N_SAMPLES": N_SAMPLES,
        "OUTLIER_CONTAMINATION": OUTLIER_CONTAMINATION,
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
    train=False,
)
dataset = mnist
# filter target labels
if VAE_TARGET_LABELS:
    indices = torch.where(
        torch.stack([(mnist.targets == t) for t in VAE_TARGET_LABELS]).sum(axis=0)
    )[0]
    dataset = torch.utils.data.Subset(mnist, indices)
# set amount of examples to be generated
if N_EXAMPLES:
    indices = torch.randperm(len(dataset))[:N_EXAMPLES]
    dataset = torch.utils.data.Subset(mnist, indices)
# create dataloader
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)


# *** Generation ***

# get means of encoded latent distributions
with torch.no_grad():
    means, labels = zip(*[(vae.encoder(x.to(device))[0], y) for x, y in dataloader])
    means = torch.cat(means, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).numpy()

unique_labels = np.unique(labels).astype("int")

np.savetxt(artifact_path / "data" / "encoded_means.txt", means)
np.savetxt(artifact_path / "data" / "encoded_labels.txt", labels)


# *** compute convex hull of means and sample from it ***

generated_z = []
generated_labels = []
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))

# build convex hulls
for i in unique_labels:
    # get points that have label i
    mask = labels == i
    points = means[mask]
    # find outliers
    outlier_mask = (
        EllipticEnvelope(contamination=OUTLIER_CONTAMINATION).fit_predict(points) == 1
        if OUTLIER_CONTAMINATION is not None
        else True
    )
    # sample new latent variables from convex hull of points
    generated_z.append(
        dist_in_hull(points[outlier_mask], n=N_SAMPLES // len(unique_labels))
    )
    generated_labels.append([i] * (N_SAMPLES // len(unique_labels)))
    # plot points reduced to 2d, with 2d convex hull
    points_2d = (
        PCA(n_components=2).fit(points).transform(points)
        if points.shape[-1] > 2
        else points
    )
    ax.scatter(*points_2d.T, s=1, color=get_cmap("tab10")(i), label=i)
    hull_2d = ConvexHull(points_2d[outlier_mask])
    for simplex in hull_2d.simplices:
        ax.plot(*points_2d[outlier_mask][simplex].T, "k-", color=get_cmap("tab10")(i))

plt.legend()
plt.savefig(artifact_path / "convex_hulls.png")
plt.close()

generated_z = Tensor(np.concatenate(generated_z, axis=0))
generated_labels = torch.tensor(
    np.concatenate(generated_labels, axis=0), dtype=torch.int32
)

np.savetxt(artifact_path / "data" / "gen_latents.txt", generated_z)
np.savetxt(artifact_path / "data" / "gen_labels.txt", generated_labels)

# plot generated latents
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
for i in unique_labels:
    mask = generated_labels == i
    z_2d = (
        PCA(n_components=2).fit(generated_z).transform(generated_z)
        if generated_z.shape[-1] > 2
        else generated_z
    )
    ax.scatter(*z_2d[mask].T, s=1, color=get_cmap("tab10")(i), label=i)
plt.legend()
plt.savefig(artifact_path / "generated_z.png")
plt.close()


# *** Generate Images from sampled latents ***

with torch.no_grad():
    x_ = vae.decoder(generated_z.to(device)).cpu()


# plot generated examples
rows, cols = len(unique_labels), 20
sorted_x_ = []
for i in unique_labels:
    mask = generated_labels == i
    sorted_x_.append(x_[mask][:cols])
sorted_x_ = reshape_to_img(
    torch.cat(sorted_x_, dim=0)[: rows * cols].numpy(), 28, 28, rows, cols
)

np.savetxt(artifact_path / "data" / "gen_img.txt", sorted_x_)
mlflow.log_image(sorted_x_, "generated_images.png")
