from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

import vae
from utils.data import ResizeDataset, get_dataset
from utils.visualization import plot_images, plot_points
from vae import VAEConfig, VAEForDataAugmentation
from vae.generation.interpolation import interpolate_along_dimension, interpolate_attribute

DATASET = "CelebA"
vae.models.base.model_store = f"pretrained_models/{DATASET}"
dataset = ResizeDataset(get_dataset(DATASET, train=False, target_type="attr"), n=50)

vae_config = VAEConfig(
    z_dim=150,
    beta=1.0,
    attr={
        "multi_vae": False,
        "mix": False,
    },
)
model = VAEForDataAugmentation.from_pretrained(vae_config, epochs=500)

latents, _, labels = model.encode_dataset(dataset).tensors
# pca = PCA(2).fit(latents) if latents.size(-1) > 2 else None
# plot_points(latents, pca, filename="img/latents.pdf")

# fakes = model.decode_dataset(TensorDataset(latents, labels)).tensors[0]
# reals = DataLoader(dataset, batch_size=50).__iter__().__next__()[0]
# plot_images(fakes, origins=reals, n=50, filename="img/fakes.pdf")
dim_latents = interpolate_along_dimension(latents[42], n_steps=10)
for i in range(dim_latents.size(0)):
    dim_fakes = model.decode_dataset(TensorDataset(dim_latents[i], torch.zeros((dim_latents[i].size(0),)))).tensors[0]
    plot_images(dim_fakes, n=10, filename=f"img/fakes-dim-{i}.pdf", cols=10)

# interpolate_attribute(latents, labels, n_steps=10)
# path = Path("~/torch_datasets/celeba/list_attr_celeba.txt").expanduser()

# attr = np.loadtxt(path, skiprows=2, usecols=list(range(1, 40)))
# print(attr.shape)
# print(attr[0])
