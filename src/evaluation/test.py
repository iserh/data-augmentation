from utils.data.split_datasets import load_splitted_datasets
import vae
from vae.models import VAEConfig
from vae.generation2 import DataAugmentation
from vae.generation import augmentations

DATASET = "MNIST"
vae.models.base.model_store = "pretrained_models/MNIST"
VAE_CONFIG = VAEConfig(z_dim=10, beta=1.0)
VAE_EPOCHS = 50
MULTI_VAE = False
SEED = 1337
K = 200

datasets, dataset_info = load_splitted_datasets(DATASET)

da = DataAugmentation(
    vae_config=VAE_CONFIG,
    vae_epochs=VAE_EPOCHS,
    multi_vae=MULTI_VAE,
    seed=SEED,
)

print(", ".join([str(len(ds)) for ds in datasets]))
print(sum([len(ds) for ds in datasets]))

augmented_datasets = da.augment_datasets(datasets, dataset_info, augmentations.REPARAMETRIZATION, K=K, balancing=False)

print(", ".join([str(len(ds)) for ds in augmented_datasets]))
print(sum([len(ds) for ds in augmented_datasets]))
