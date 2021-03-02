import mlflow
import torch
import vae
from vae import VAEConfig, train_vae
from vae.models.architectures import VAEModelV2 as VAEModelVersion
from utils.data import load_splitted_datasets, get_dataset, BatchDataset
from utils.mlflow import backend_stores
from utils.trainer import TrainingArguments

# *** Seeding, loading data & setting up mlflow logging ***

DATASET = "CelebA"
SEED = 1337

# set the backend store uri of mlflow
mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
# seed torch
torch.manual_seed(SEED)

# *** VAE Parameters ***

MULTI_VAE = False
VAE_EPOCHS = 500
Z_DIM = 100
BETA = 1.0
MIX = False

# *** Training the VAE ***

# set mlflow experiment
mlflow.set_experiment(f"Z_DIM {Z_DIM}")

if MULTI_VAE:
    vae.models.base.model_store = f"pretrained_models/{DATASET}/MULTI/{'MIX' if MIX else '~MIX'}/Z_DIM {Z_DIM}"
    datasets, dataset_info = load_splitted_datasets(DATASET, others=MIX)
    for label, train_dataset, class_count in zip(dataset_info["classes"], datasets, dataset_info["class_counts"]):
        with mlflow.start_run():
            mlflow.log_param("class_count", class_count)
            train_vae(
                TrainingArguments(VAE_EPOCHS, batch_size=128),
                train_dataset=BatchDataset(train_dataset, 100 * 128),
                model_architecture=VAEModelVersion,
                vae_config=VAEConfig(z_dim=Z_DIM, beta=BETA, attr={"label": label}),
                save_every_n_epochs=50,
                seed=SEED,
            )
else:
    vae.models.base.model_store = f"pretrained_models/MNIST/SINGLE/Z_DIM {Z_DIM}"
    train_dataset = get_dataset(DATASET, train=True)
    with mlflow.start_run():
        train_vae(
            TrainingArguments(VAE_EPOCHS, batch_size=128),
            train_dataset=train_dataset,
            model_architecture=VAEModelVersion,
            vae_config=VAEConfig(z_dim=Z_DIM, beta=BETA),
            save_every_n_epochs=100,
            seed=SEED,
        )
