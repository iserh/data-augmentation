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
experiment = mlflow.get_experiment_by_name("CelebA Generation")
if not experiment:
    experiment = mlflow.get_experiment(mlflow.create_experiment("CelebA Generation"))
run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name="standard")
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

# Load dataset
celeba = CelebA(
    root="~/torch_datasets",
    transform=transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    split="test",
    target_type="identity",
    download=False,
)
celeba = [celeba[256]]
print(f"Dataset size: {len(celeba)}")
dataloader = DataLoader(
    celeba,
    batch_size=512,
    shuffle=True,
    num_workers=4,
)


# *** Generation ***

print("Encoding dataset...")
# get means of encoded latent distributions
with torch.no_grad():
    m_v_log, labels = zip(*[(vae.encoder(x.to(device)), y) for x, y in dataloader])
    means, variance_logs = zip(*m_v_log)
    means = torch.cat(means, dim=0).cpu()
    variance_logs = torch.cat(variance_logs, dim=0).cpu()
    labels = torch.cat(labels, dim=0).numpy()
print("Encoding done!")

print("Sampling from datasets")
# extend / shrink dataset to N_SAMPLES
if N_SAMPLES:
    random_indices = torch.randint(0, means.size(0), size=(N_SAMPLES,))
else:
    random_indices = torch.randperm(means.size(0))
means = means[random_indices]
variance_logs = variance_logs[random_indices]
generated_labels = labels[random_indices]
unique_labels = np.unique(generated_labels).astype("int")

# run pca for visualization
pca_2d = PCA(n_components=2).fit(means)

# generate new latents
eps = torch.empty_like(variance_logs).normal_()
generated_z = eps * (0.5 * variance_logs).exp() + means

# for visualization
generated_z_2d = pca_2d.transform(generated_z)

# plot generated latents
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
for i in unique_labels:
    mask = generated_labels == i
    ax.scatter(*generated_z_2d[mask].T, s=1, color=get_cmap("tab10")(i), label=i)
plt.legend()
plt.savefig(artifact_path / "generated_z.png")
plt.close()

# generate images
with torch.no_grad():
    x_ = vae.decoder(generated_z.to(device))


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(x_[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig(artifact_path / "fake_real.png")

finished = True
