"""Evaluate the performance of augmentation on CNNs."""
import mlflow

from utils.mlflow import backend_stores
from utils.trainer import TrainingArguments
from vae.models import VAEConfig
from vae import train_vae_on_dataset, train_vae_on_classes

from evaluation.create_datasets import DATASET, load_datasets

# Parameter for classification task
SEED = 1337
# set the backend store uri of mlflow
mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
# load datasets
train_dataset, vae_train_dataset, val_dataset, test_dataset = load_datasets()

VAE_EPOCHS = 50
for z_dim in [10]:
    for beta in [1.0]:
        mlflow.set_experiment(f"Z_DIM {z_dim}")
        vae_config = VAEConfig(z_dim=z_dim, beta=beta)
        train_vae_on_classes(
            training_args=TrainingArguments(VAE_EPOCHS, seed=SEED, batch_size=64),
            train_dataset=vae_train_dataset,
            test_dataset=val_dataset,
            vae_config=vae_config,
            save_every_n_epochs=10,
            seed=SEED,
        )
    
# VAE_EPOCHS = 100
# for z_dim in [2, 10, 20, 50, 100]:
#     for beta in [0.3, 0.5, 0.7, 1.0]:
#         mlflow.set_experiment(f"Z_DIM {z_dim}")
#         vae_config = VAEConfig(z_dim=z_dim, beta=beta)
#         train_vae_on_dataset(
#             training_args=TrainingArguments(VAE_EPOCHS, seed=SEED, batch_size=128),
#             train_dataset=vae_train_dataset,
#             test_dataset=val_dataset,
#             vae_config=vae_config,
#             save_every_n_epochs=25,
#             seed=SEED,
#         )
