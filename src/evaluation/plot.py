from pathlib import Path

import vae
from utils.visualization import plot_images, plot_points
from vae import VAEConfig, VAEForDataAugmentation

import vae
from utils.data import load_unsplitted_dataset
from vae import VAEConfig
import torch

# *** Loading data ***

SEED = 1337
DATASET = "MNIST"
MIX = False
# load datasets
train_dataset, dataset_info = load_unsplitted_dataset(DATASET)

# *** Parameters for data augmentation ***

MULTI_VAE = True
vae_config = VAEConfig(
    z_dim=2,
    beta=1.0,
    label=3,
    attr={
        "mix": False,
        "seed": SEED,
    },
)
VAE_EPOCHS = 20
# set model store path
if MULTI_VAE:
    vae.models.base.model_store = f"pretrained_models/{DATASET}/{sum(dataset_info['class_counts'])}"
    # vae.models.base.model_store = f"pretrained_models/{DATASET}/60000"
else:
    vae.models.base.model_store = f"pretrained_models/{DATASET}/all_data_single"

model = VAEForDataAugmentation.from_pretrained(vae_config, VAE_EPOCHS)

mean = torch.zeros(size=(100, vae_config.z_dim))
std = torch.zeros_like(mean) + 1
normal = torch.normal(mean, std)
labels = torch.zeros(size=(100,))
decoded = model.decode_dataset(torch.utils.data.TensorDataset(normal, labels)).tensors[0]
plot_images(decoded, n=50, filename=f"img/label-{vae_config.label}-epoch-{VAE_EPOCHS}.pdf")
