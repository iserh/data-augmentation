import mlflow
import torch
import vae
from vae import VAEConfig, train_vae
from vae.models.architectures import VAEModelV1 as VAEModelVersion
from utils.data import load_splitted_datasets, get_dataset, BatchDataset
from utils.mlflow import backend_stores
from utils.trainer import TrainingArguments

# *** Seeding, loading data & setting up mlflow logging ***

DATASET = "MNIST"
SEED = 42

# set the backend store uri of mlflow
mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
# seed torch
torch.manual_seed(SEED)

# *** VAE Parameters ***

vae_config = VAEConfig(z_dim=8, beta=1.0, attr={
    "multi_vae": True,
    "mix": False,
})

# *** Training the VAE ***

training_args = TrainingArguments(
    epochs=2500,
    batch_size=64,
    save_epochs=500,
    log_steps=50,
    num_workers=4,
    lr=5e-4
)

# set mlflow experiment
mlflow.set_experiment(f"Z_DIM {vae_config.z_dim}")

if vae_config.attr["multi_vae"]:
    datasets, dataset_info = load_splitted_datasets(DATASET, others=vae_config.attr["mix"])
    vae.models.base.model_store = f"pretrained_models/{DATASET}/{sum(dataset_info['class_counts'])}"
    for label, train_dataset, class_count in zip(dataset_info["classes"], datasets, dataset_info["class_counts"]):
        with mlflow.start_run():
            mlflow.log_param("class_count", class_count)
            vae_config.attr["label"] = label
            train_vae(
                training_args=training_args,
                train_dataset=train_dataset,
                model_architecture=VAEModelVersion,
                vae_config=vae_config,
                seed=SEED,
            )
else:
    train_dataset = get_dataset(DATASET, train=True)
    with mlflow.start_run():
        train_vae(
            training_args=training_args,
            train_dataset=train_dataset,
            model_architecture=VAEModelVersion,
            vae_config=vae_config,
            seed=SEED,
        )
