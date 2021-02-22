"""Evaluate the performance of augmentation on CNNs."""
import mlflow
import torch

from utils.mlflow import backend_stores
from utils.trainer import TrainingArguments
from vae.generation import augment_dataset_using_per_class_vaes, augment_dataset_using_single_vae, augmentations
from vae.models import VAEConfig

from evaluation.models import CNNMNIST
from evaluation.train_model import train_model
from evaluation.create_datasets import DATASET, DATASET_LIMIT, load_datasets
import toml

# Parameter for classification task
SEED = 1337
pyproject = toml.load("pyproject.toml")
training_args = TrainingArguments(
    total_steps=5000,
    batch_size=32,
    validation_intervall=200,
    save_model=False,
    seed=SEED,
    early_stopping=False,
    early_stopping_window=20,
    save_best_metric="best_acc",
)
# Parameter for augmentation task
PER_CLASS_VAE = False
AUGMENTATION = augmentations.FORWARD
augmentation_params = {"k": 9}

# set the backend store uri of mlflow
mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
# seed torch
torch.manual_seed(SEED)

# load datasets
train_dataset, vae_train_dataset, val_dataset, test_dataset = load_datasets()

pyproject["project"]["paths"]["model_root"] = "pretrained_models/MNIST/SINGLE"
with open("pyproject.toml", "w") as f:
    toml.dump(pyproject, f)

for z_dim in [2, 10, 20, 50, 100]:
    for beta in [1.0]:
        for VAE_EPOCHS in [25, 50, 100]:
            vae_config = VAEConfig(z_dim=z_dim, beta=beta)
            # start mlflow run in experiment
            mlflow.set_experiment("CNN Training")
            with mlflow.start_run(run_name=AUGMENTATION or "baseline"):
                mlflow.log_param("dataset_limit", DATASET_LIMIT)
                # perform data augmentation
                if AUGMENTATION is not None:
                    mlflow.log_param("per_class_vae", PER_CLASS_VAE)
                    mlflow.log_param("vae_epochs", VAE_EPOCHS)
                    if PER_CLASS_VAE:
                        train_dataset = augment_dataset_using_per_class_vaes(
                            train_dataset, vae_config, VAE_EPOCHS, AUGMENTATION, augmentation_params, seed=SEED
                        )
                    else:
                        train_dataset = augment_dataset_using_single_vae(
                            train_dataset, vae_config, VAE_EPOCHS, AUGMENTATION, augmentation_params, seed=SEED
                        )

                # train cnn
                results = train_model(
                    model=CNNMNIST(),
                    training_args=training_args,
                    train_dataset=train_dataset,
                    dev_dataset=val_dataset,
                    test_dataset=test_dataset,
                    seed=SEED,
                )

                print(results)
