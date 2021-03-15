import mlflow
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

import vae
from utils.data import BatchDataset, get_dataset, load_splitted_datasets
from utils.data.split_datasets import load_unsplitted_dataset
from utils.mlflow import backend_stores
from utils.trainer import TrainingArguments
from vae import VAEConfig, VAEForDataAugmentation

# *** Seeding, loading data & setting up mlflow logging ***

DATASET = "thyroid"
SEED = 1337

# set the backend store uri of mlflow
mlflow.set_tracking_uri(getattr(backend_stores, DATASET))

# *** VAE Parameters ***

vae_config = VAEConfig(z_dim=2, beta=1.0)
vae.models.base.model_store = f"pretrained_models/{DATASET}/SINGLE/Z_DIM {vae_config.z_dim}"
model = VAEForDataAugmentation.from_pretrained(vae_config, epochs=100)

# set mlflow experiment
mlflow.set_experiment(f"Z_DIM {vae_config.z_dim}")

dataset, dataset_info = load_unsplitted_dataset("thyroid")

encoded_tensors = model.encode_dataset(dataset).tensors
reconstructed = model.decode_dataset(TensorDataset(encoded_tensors[0], encoded_tensors[2])).tensors[0]

reals = next(iter(DataLoader(dataset, batch_size=len(dataset))))[0]

fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].bar(range(reals.size(1)), reals[0, :])
ax[1].bar(range(reconstructed.size(1)), reconstructed[0, :])
plt.savefig("pic.png")
plt.close()
