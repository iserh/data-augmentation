"""Data generation with convex hulls."""
from utils import get_artifact_path, reshape_to_img
from vae.model_setup import load_model

import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

# *** HYPERPARAMETERS ***

EPOCHS = 20
Z_DIM = 2
ALPHA = 1.0
BETA = 1.0

TARGET_LABEL = 4
N_SAMPLES = 2_000

# initialize mlflow experiment & run
experiment = mlflow.get_experiment_by_name("MNIST Generation")
if not experiment:
    experiment = mlflow.get_experiment(mlflow.create_experiment("MNIST Generation"))
run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name="standard")
artifact_path = get_artifact_path(run)
(artifact_path / "data").mkdir(exist_ok=True, parents=True)

mlflow.log_params(
    {
        "epochs": EPOCHS,
        "z_dim": Z_DIM,
        "alpha": ALPHA,
        "beta": BETA,
        "n_samples": N_SAMPLES,
        "target_label": TARGET_LABEL,
    }
)


# *** Data preparation ***

(vae, device), _ = load_model(EPOCHS, Z_DIM, ALPHA, BETA, TARGET_LABEL, cuda=False)

# Load dataset
mnist = MNIST(
    root="~/torch_datasets",
    download=True,
    transform=ToTensor(),
    train=False,
)
if TARGET_LABEL is not None:
    target_idx = torch.where(mnist.targets == TARGET_LABEL)[0]
    dataset = torch.utils.data.Subset(
        mnist, target_idx[torch.randint(0, target_idx.size(0), (N_SAMPLES,))]
    )
else:
    dataset = torch.utils.data.Subset(
        mnist, torch.randint(0, len(mnist), (N_SAMPLES,))
    )
dataloader = DataLoader(dataset, batch_size=512, shuffle=False)


# *** Generation ***

# get means of encoded latent distributions
with torch.no_grad():
    m_v_log, labels = zip(*[(vae.encoder(x.to(device)), y) for x, y in dataloader])
    means, variance_logs = zip(*m_v_log)
    means = torch.cat(means, dim=0).cpu()
    variance_logs = torch.cat(variance_logs, dim=0).cpu()
    labels = torch.cat(labels, dim=0)

unique_labels = torch.unique(labels, sorted=True).int().tolist()

np.savetxt(artifact_path / "data" / "encoded_means.txt", means.numpy())
np.savetxt(artifact_path / "data" / "encoded_labels.txt", labels.numpy())

# extend / shrink dataset to N_SAMPLES
random_idx = torch.randint(0, means.size(0), size=(N_SAMPLES,))
means = means[random_idx]
variance_logs = variance_logs[random_idx]
generated_labels = labels[random_idx]

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
