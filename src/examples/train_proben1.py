from typing import Dict
import mlflow
import torch
import vae
import generative_classifier
from utils.mlflow import backend_stores
from utils.trainer import TrainingArguments, Trainer
from vae import VAEConfig, augmentations, DataAugmentation
from evaluation import ModelProben1
from utils.data import get_dataset, load_splitted_datasets, load_unsplitted_dataset
from torch.utils.data import ConcatDataset
from utils.models import ModelConfig
from torch import Tensor
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(predictions: Tensor, labels: Tensor) -> Dict[str, float]:
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"acc": acc, "f1": f1}


# *** Seeding, loading data & setting up mlflow logging ***

SEED = 1337
DATASET = "diabetes"
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
    z_dim=2,
    beta=0.001,
    attr={
        "mix": True,
        "multi_vae": True,
    }
)
VAE_EPOCHS = 625
AUGMENTATION = augmentations.NORMAL_NOISE
USE_GC = False
K = 400
augmentation_params = {"std": 1}
# set model store path
vae.models.base.model_store = f"pretrained_models/{DATASET}"
# set gc path
generative_classifier.models.base.model_store = "generative_classifiers/{DATASET}"

# *** Data Augmentation ***

mlflow.set_experiment(f"CNN Z_DIM {vae_config.z_dim}" if AUGMENTATION else "CNN Baseline")
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
model_cfg = ModelConfig(attr={
    "in_feat": 8,
    "out_feat": 2,
    "N": 128,
    "M": 256,
    "K": 512,
})

# *** Training the CNN ***

# start mlflow run in experiment
with mlflow.start_run(run.info.run_id):
    # seeding
    if SEED is not None:
        torch.manual_seed(SEED)
    # trainer
    trainer = Trainer(
        args=training_args,
        model=ModelProben1(model_cfg),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    # train model
    trainer.train()
    # evaluate model
    print(trainer.evaluate())
