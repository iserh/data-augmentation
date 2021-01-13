"""Data generation with convex hulls."""
import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from utils import get_artifact_path, reshape_to_img
from vae.model_setup import load_model

# *** HYPERPARAMETERS ***

VAE_EPOCHS = 50
VAE_Z_DIM = 10
VAE_ALPHA = 1.0
VAE_BETA = 1.0
VAE_N_EXAMPLES_LIMIT = 500
VAE_TARGET_LABELS = None

N_SAMPLES = None


# *** Mlflow initialization ***

# initialize mlflow experiment & run
experiment = mlflow.get_experiment_by_name("MNIST Generation")
if not experiment:
    experiment = mlflow.get_experiment(mlflow.create_experiment("MNIST Generation"))
run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name="standard")
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
        "N_SAMPLES": N_SAMPLES,
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
dataloader = DataLoader(dataset, batch_size=512, shuffle=False)


# *** Generation ***

# get means of encoded latent distributions
with torch.no_grad():
    m_v_log, labels = zip(*[(vae.encoder(x.to(device)), y) for x, y in dataloader])
    means, variance_logs = zip(*m_v_log)
    means = torch.cat(means, dim=0).cpu()
    variance_logs = torch.cat(variance_logs, dim=0).cpu()
    labels = torch.cat(labels, dim=0).numpy()


np.savetxt(artifact_path / "data" / "encoded_means.txt", means.numpy())
np.savetxt(artifact_path / "data" / "encoded_labels.txt", labels)

# extend / shrink dataset to N_SAMPLES
if N_SAMPLES:
    random_indices = torch.randint(0, means.size(0), size=(N_SAMPLES,))
else:
    random_indices = torch.randperm(means.size(0))
means = means[random_indices]
variance_logs = variance_logs[random_indices]
generated_labels = labels[random_indices]
unique_labels = np.unique(generated_labels).astype("int")

np.savetxt(artifact_path / "data" / "gen_labels.txt", generated_labels)

# generate new latents
eps = torch.empty_like(variance_logs).normal_()
generated_z = eps * (0.5 * variance_logs).exp() + means

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

# generate images
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
