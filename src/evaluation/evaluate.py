from typing import Dict

import mlflow
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from torch.utils.data import ConcatDataset

import vae
from utils.data import get_dataset, load_splitted_datasets, load_unsplitted_dataset
from utils.mlflow import backend_stores
from utils.models import ModelConfig
from utils.trainer import Trainer, TrainingArguments
from vae import DataAugmentation, VAEConfig, augmentations

import generative_classifier
from evaluation import CNNMNIST as ModelForClassification
# from evaluation import ModelProben1 as ModelForClassification


def compute_metrics(predictions: Tensor, labels: Tensor) -> Dict[str, float]:
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"acc": acc, "f1": f1}


# *** Seeding, loading data & setting up mlflow logging ***

SEED = 42
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

vae_config = VAEConfig(
    z_dim=3,
    beta=1.0,
    attr={
        "mix": False,
        "multi_vae": True,
    },
)
VAE_EPOCHS = 3000
AUGMENTATION = augmentations.REPARAMETRIZATION
USE_GC = False
K = 500
augmentation_params = {}
# set model store path
vae.models.base.model_store = f"pretrained_models/{DATASET}/{sum(dataset_info['class_counts'])}"
# set gc path
# generative_classifier.models.base.model_store = f"generative_classifiers/{DATASET}"

# *** Data Augmentation ***

mlflow.set_experiment(f"Evaluation {'MULTI' if vae_config.attr['multi_vae'] else 'SINGLE'} {sum(dataset_info['class_counts'])}")
with mlflow.start_run(run_name=AUGMENTATION or "baseline") as run:
    mlflow.log_param("original_dataset_size", len(train_dataset))
    if AUGMENTATION is not None:
        da = DataAugmentation(
            vae_config=vae_config,
            vae_epochs=VAE_EPOCHS,
            gc_config=ModelConfig() if USE_GC else None,
            gc_epochs=1,
            multi_vae=vae_config.attr["multi_vae"],
            seed=SEED,
        )

        generated_datasets = da.augment_datasets(datasets, dataset_info, AUGMENTATION, K=K, **augmentation_params)
        train_dataset = ConcatDataset([train_dataset, *generated_datasets])

test_dataset = get_dataset(DATASET, train=False)

# *** The parameters for the classification task ***

training_args = TrainingArguments(
    epochs=50,
    batch_size=32,
    save_model=False,
    seed=SEED,
    metric_for_best_model="acc",
    log_steps=None,
)
# PROBEN1
# model_cfg = ModelConfig(
#     attr={
#         "in_feat": 8,
#         "out_feat": 2,
#         "N": 128,
#         "M": 256,
#         "K": 512,
#     }
# )
# MNIST
model_cfg = ModelConfig()

# *** Training the CNN ***

# start mlflow run in experiment
with mlflow.start_run(run.info.run_id):
    # seeding
    if SEED is not None:
        torch.manual_seed(SEED)
    # trainer
    trainer = Trainer(
        args=training_args,
        model=ModelForClassification(model_cfg),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    # train model
    trainer.train()
    # evaluate model
    print(trainer.evaluate())
