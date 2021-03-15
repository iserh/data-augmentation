import mlflow
import torch
from torch.utils.data import ConcatDataset

import vae
from utils.data import BatchDataset, get_dataset, load_splitted_datasets, load_unsplitted_dataset
from utils.data.dataset import ResizeDataset
from utils.mlflow import backend_stores
from utils.models import ModelConfig
from utils.trainer import TrainingArguments
from vae import DataAugmentation, VAEConfig, augmentations

import generative_classifier
from evaluation import CNNMNIST

# *** Seeding, loading data & setting up mlflow logging ***

SEED = 1337
DATASET = "CelebA"
MIX = False
# seed torch
torch.manual_seed(SEED)
# set the backend store uri of mlflow
mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
# load datasets
# datasets, dataset_info = load_splitted_datasets(DATASET, others=MIX)
# train_dataset, _ = load_unsplitted_dataset(DATASET)
train_dataset = ResizeDataset(get_dataset(DATASET, train=False), n=500)

# *** Parameters for data augmentation ***

MULTI_VAE = False
VAE_EPOCHS = 500
Z_DIM = 150
BETA = 1.0
AUGMENTATION = augmentations.REPARAMETRIZATION
K = 1000
augmentation_params = {}
# set model store path
vae.models.base.model_store = f"pretrained_models/{DATASET}"
# set gc path
# generative_classifier.models.base.model_store = "generative_classifiers/{DATASET}"

# *** Data Augmentation ***

mlflow.set_experiment(f"CNN Z_DIM {Z_DIM}" if AUGMENTATION else "CNN Baseline")
with mlflow.start_run(run_name=AUGMENTATION or "baseline") as run:
    mlflow.log_param("original_dataset_size", len(train_dataset))
    if AUGMENTATION is not None:
        da = DataAugmentation(
            vae_config=VAEConfig(z_dim=Z_DIM, beta=BETA, attr={"mix": MIX, "multi_vae": MULTI_VAE}),
            vae_epochs=VAE_EPOCHS,
            gc_config=None,
            gc_epochs=1,
            multi_vae=MULTI_VAE,
            seed=SEED,
        )

        generated_datasets = da.augment_datasets(
            [train_dataset],
            {"n_classes": 1, "class_counts": [len(train_dataset)], "classes": [0]},
            AUGMENTATION,
            K=K,
            **augmentation_params,
        )
        train_dataset = ConcatDataset([train_dataset, *generated_datasets])

test_dataset = get_dataset(DATASET, train=False)

# # *** The parameters for the classification task ***

# training_args = TrainingArguments(
#     total_steps=5000,
#     batch_size=32,
#     validation_intervall=200,
#     save_model=False,
#     seed=SEED,
#     early_stopping=False,
#     early_stopping_window=20,
#     metric_for_best_model="best_acc",
# )

# # *** Training the CNN ***

# # create a vae config
# vae_config = VAEConfig(z_dim=Z_DIM, beta=BETA)
# # start mlflow run in experiment
# with mlflow.start_run(run.info.run_id):
#     # train cnn
#     results = train_model(
#         model=CNNMNIST(),
#         training_args=training_args,
#         train_dataset=BatchDataset(train_dataset, 100 * training_args.batch_size),
#         dev_dataset=test_dataset,
#         test_dataset=test_dataset,
#         seed=SEED,
#     )
#     # print the results
#     print(results)
