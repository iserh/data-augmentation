import mlflow
import torch

import vae
from utils.data import BatchDataset, get_dataset, load_splitted_datasets, load_unsplitted_dataset
from utils.mlflow import backend_stores
from utils.trainer import TrainingArguments
from utils import seed_everything
from vae import VAEConfig, train_vae
from vae.models.architectures import VAEModelV1 as VAEModelVersion

# *** Seeding, loading data & setting up mlflow logging ***

DATASET = "MNIST"
SEED = 1337

# set the backend store uri of mlflow
mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
# seed torch
seed_everything(SEED)

# *** VAE Parameters ***

MULTI_VAE = True
vae_config = VAEConfig(
    z_dim=2,
    beta=1.0,
    attr={
        "mix": False,
        "seed": SEED,
    },
)

# *** Training the VAE ***

training_args = TrainingArguments(epochs=20, batch_size=64, save_epochs=2, num_workers=0, lr=5e-5, weight_decay=3e-5, seed=SEED)

# set mlflow experiment

if MULTI_VAE:
    # load class datasets
    datasets, dataset_info = load_splitted_datasets(DATASET, others=vae_config.attr["mix"])
    # set the location of the saved model
    vae.models.base.model_store = f"pretrained_models/{DATASET}/{sum(dataset_info['class_counts'])}"
    # set mlflow experiment
    mlflow.set_experiment(f"{sum(dataset_info['class_counts'])} MULTI Z_DIM {vae_config.z_dim}")
    # train a vae for each class
    for label, train_dataset, class_count in zip(dataset_info["classes"], datasets, dataset_info["class_counts"]):
        # artificially increase dataset size
        train_dataset = BatchDataset(train_dataset, training_args.batch_size * 100)
        # set the label of the vae
        vae_config.label = label

        with mlflow.start_run():
            mlflow.log_param("original_dataset_size", len(train_dataset))
            train_vae(
                training_args=training_args,
                train_dataset=train_dataset,
                model_architecture=VAEModelVersion,
                vae_config=vae_config,
                seed=SEED,
            )
else:
    # load dataset
    train_dataset, dataset_info = load_unsplitted_dataset(DATASET)
    # train_dataset = get_dataset(DATASET, train=True)

    # set mlflow experiment
    mlflow.set_experiment(f"{len(train_dataset)} SINGLE")
    # set the location of the saved model
    vae.models.base.model_store = f"pretrained_models/{DATASET}/{len(train_dataset)} SINGLE"

    train_dataset = BatchDataset(train_dataset, training_args.batch_size * 100)

    with mlflow.start_run():
        train_vae(
            training_args=training_args,
            train_dataset=train_dataset,
            model_architecture=VAEModelVersion,
            vae_config=vae_config,
            seed=SEED,
        )
