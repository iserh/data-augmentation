"""Data generation with convex hulls."""
import mlflow
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from utils import get_artifact_path
from vae.model_setup import load_model

# *** HYPERPARAMETERS ***

EPOCHS = 20
Z_DIM = 2
ALPHA = 1.0
BETA = 1.0

N_EXAMPLES = 10_000
TARGET_LABEL = 4


# *** Mlflow initialization ***

# initialize mlflow experiment & run
experiment = mlflow.get_experiment_by_name("Feature Space")
if not experiment:
    experiment = mlflow.get_experiment(mlflow.create_experiment("Feature Space"))
run = mlflow.start_run(experiment_id=experiment.experiment_id, run_name="encoded_space")
artifact_path = get_artifact_path(run)
(artifact_path / "data").mkdir(exist_ok=True, parents=True)

# log hyperparameters
mlflow.log_params(
    {
        "epochs": EPOCHS,
        "z_dim": Z_DIM,
        "alpha": ALPHA,
        "beta": BETA,
        "n_examples": N_EXAMPLES,
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
        mnist, target_idx[torch.randint(0, target_idx.size(0), (N_EXAMPLES,))]
    )
else:
    dataset = torch.utils.data.Subset(
        mnist, torch.randint(0, len(mnist), (N_EXAMPLES,))
    )
dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

# *** Encoder Space Visualization ***

# get means of encoded latent distributions
with torch.no_grad():
    m_v_log, labels = zip(*[(vae.encoder(x.to(device)), y) for x, y in dataloader])
    means, variance_logs = zip(*m_v_log)
    means = torch.cat(means, dim=0).cpu().numpy()
    variance_logs = torch.cat(variance_logs, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0)

unique_labels = torch.unique(labels, sorted=True).int().tolist()

# plot encoded latent means
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
for i in unique_labels:
    mask = labels == i
    z_2d = (
        PCA(n_components=2).fit(means).transform(means)
        if means.shape[-1] > 2
        else means
    )
    ax.scatter(*z_2d[mask].T, s=1, color=get_cmap("tab10")(i), label=i)
plt.legend()
plt.savefig(artifact_path / "generated_z.png")
plt.close()
