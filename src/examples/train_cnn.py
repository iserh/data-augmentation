import mlflow
import torch
import vae
import generative_classifier
from utils.mlflow import backend_stores
from utils.trainer import TrainingArguments
from vae import VAEConfig, augmentations, DataAugmentation
from evaluation import train_model, CNNMNIST
from utils.data import get_dataset, load_splitted_datasets, load_unsplitted_dataset, BatchDataset
from torch.utils.data import ConcatDataset
from utils.models import ModelConfig

# *** Seeding, loading data & setting up mlflow logging ***

SEED = 1337
DATASET = "MNIST"
MIX = False
# seed torch
torch.manual_seed(SEED)
# set the backend store uri of mlflow
mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
# load datasets
datasets, dataset_info = load_splitted_datasets(DATASET, others=MIX)
train_dataset, _ = load_unsplitted_dataset(DATASET)

# *** Parameters for data augmentation ***

MULTI_VAE = False
VAE_EPOCHS = 100
Z_DIM = 10
BETA = 1.0
AUGMENTATION = augmentations.REPARAMETRIZATION
K = 450
augmentation_params = {}
# set model store path
append_to_path = ("MULTI/" + ("MIX" if MIX else "~MIX")) if MULTI_VAE else "SINGLE"
vae.models.base.model_store = f"pretrained_models/MNIST/{append_to_path}/Z_DIM {Z_DIM}"
# set gc path
generative_classifier.models.base.model_store = "generative_classifiers/MNIST"

# *** Data Augmentation ***

mlflow.set_experiment(f"CNN Z_DIM {Z_DIM}" if AUGMENTATION else "CNN Baseline")
with mlflow.start_run(run_name=AUGMENTATION or "baseline") as run:
    mlflow.log_param("original_dataset_size", len(train_dataset))
    if AUGMENTATION is not None:
        da = DataAugmentation(
            vae_config=VAEConfig(z_dim=Z_DIM, beta=BETA),
            vae_epochs=VAE_EPOCHS,
            gc_config=ModelConfig(),
            gc_epochs=1,
            multi_vae=MULTI_VAE,
            seed=SEED,
        )

        generated_datasets = da.augment_datasets(datasets, dataset_info, AUGMENTATION, K=K, **augmentation_params)
        train_dataset = ConcatDataset([train_dataset, *generated_datasets])

test_dataset = get_dataset(DATASET, train=False)

# *** The parameters for the classification task ***

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

# *** Training the CNN ***

# create a vae config
vae_config = VAEConfig(z_dim=Z_DIM, beta=BETA)
# start mlflow run in experiment
with mlflow.start_run(run.info.run_id):
    # train cnn
    results = train_model(
        model=CNNMNIST(),
        training_args=training_args,
        train_dataset=BatchDataset(train_dataset, 100 * training_args.batch_size),
        dev_dataset=test_dataset,
        test_dataset=test_dataset,
        seed=SEED,
    )
    # print the results
    print(results)
