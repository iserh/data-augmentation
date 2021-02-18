"""Evaluate the performance of augmentation on CNNs."""
import mlflow
import torch
from torch.utils.data import random_split

from utils.data import get_dataset
from utils.mlflow import backend_stores
from utils.trainer import TrainingArguments
from vae.generation import augment_dataset_using_per_class_vaes, augment_dataset_using_single_vae, augmentations
from vae.models import VAEConfig

from evaluation.train_model import train_model

# Parameter for classification task
DATASET = "MNIST"
DATASET_LIMIT = 50
training_args = TrainingArguments(epochs=10, save_intervall=None, save_model=False, seed=1337)
# Parameter for augmentation task
PER_CLASS_VAE = False
VAE_EPOCHS = 25
vae_config = VAEConfig(z_dim=10, beta=1.0)
AUGMENTATION = augmentations.FORWARD
augmentation_params = {"k": 100}

# set the backend store uri of mlflow
mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
# seed torch
torch.manual_seed(1337)

# load test dataset
test_dataset = get_dataset(DATASET, train=False)
# load train dataset
train_dataset = get_dataset(DATASET, train=True)
# limit train dataset corresponding to DATASET_LIMIT and add 5000 for dev dataset
dev_test_size = 5000 // (augmentation_params["k"] * augmentation_params.get("k2", 1) + 1)
train_dataset, dev_dataset, _ = random_split(
    train_dataset, [DATASET_LIMIT, dev_test_size, len(train_dataset) - DATASET_LIMIT - dev_test_size]
)

# start mlflow run in experiment
mlflow.set_experiment("CNN Training")
with mlflow.start_run(run_name=AUGMENTATION or "baseline"):
    mlflow.log_param("dataset_limit", DATASET_LIMIT)
    # perform data augmentation
    if AUGMENTATION is not None:
        if PER_CLASS_VAE:
            dev_dataset = augment_dataset_using_per_class_vaes(
                dev_dataset, vae_config, VAE_EPOCHS, AUGMENTATION, augmentation_params, seed=1337
            )
            train_dataset = augment_dataset_using_per_class_vaes(
                train_dataset, vae_config, VAE_EPOCHS, AUGMENTATION, augmentation_params, seed=1337
            )
        else:
            dev_dataset = augment_dataset_using_single_vae(
                dev_dataset, vae_config, VAE_EPOCHS, AUGMENTATION, augmentation_params, seed=1337
            )
            train_dataset = augment_dataset_using_single_vae(
                train_dataset, vae_config, VAE_EPOCHS, AUGMENTATION, augmentation_params, seed=1337
            )

    # train cnn
    results = train_model(
        training_args=training_args,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        seed=1337,
    )

    print(results)
