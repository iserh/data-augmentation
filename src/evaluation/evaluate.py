import mlflow
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import ConcatDataset

import vae
from utils.data import get_dataset, load_splitted_datasets, load_unsplitted_dataset, BatchDataset
from utils.mlflow import backend_stores
from utils.models import ModelConfig
from utils.trainer import Trainer, TrainingArguments
from utils import seed_everything
from vae import DataAugmentation, VAEConfig, augmentations

import generative_classifier
from evaluation import CNNMNIST as ModelForClassification
# from evaluation import ModelProben1 as ModelForClassification

# *** Loading data & setting up mlflow logging ***

SEED = 1337
DATASET = "MNIST"
MIX = False
# set the backend store uri of mlflow
mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
# load datasets
datasets, dataset_info = load_splitted_datasets(DATASET, others=MIX)
train_dataset, _ = load_unsplitted_dataset(DATASET)

# *** Parameters for data augmentation ***

MULTI_VAE = True
vae_config = VAEConfig(
    z_dim=2,
    beta=1.0,
    attr={
        "mix": False,
        "seed": SEED,
    },
)
VAE_EPOCHS = 20
AUGMENTATION = None
USE_GC = False
K = 5000
augmentation_params = {}
# set model store path
vae.models.base.model_store = f"pretrained_models/{DATASET}/{sum(dataset_info['class_counts'])}{'' if MULTI_VAE else ' SINGLE'}"
# set gc path
# generative_classifier.models.base.model_store = f"generative_classifiers/{DATASET}"

# set mlflow experiment
mlflow.set_experiment(f"{sum(dataset_info['class_counts'])} {'MULTI' if MULTI_VAE else 'SINGLE'} Evaluation")

# *** Data Augmentation ***

# seed torch
with mlflow.start_run(run_name=AUGMENTATION or "baseline") as run:
    mlflow.log_param("original_dataset_size", len(train_dataset))
    if AUGMENTATION is not None:
        da = DataAugmentation(
            vae_config=vae_config,
            vae_epochs=VAE_EPOCHS,
            gc_config=ModelConfig() if USE_GC else None,
            gc_epochs=1,
            multi_vae=MULTI_VAE,
            seed=SEED,
        )

        generated_datasets = da.augment_datasets(datasets, dataset_info, AUGMENTATION, K=K, **augmentation_params)
        train_dataset = ConcatDataset([*generated_datasets])

test_dataset = get_dataset(DATASET, train=False)

# *** The parameters for the classification task ***

training_args = TrainingArguments(
    epochs=10,
    save_model=False,
    seed=SEED,
    batch_size=32,
    metric_for_best_model="f1",
    lr=5e-3,
    num_workers=0.
)

# model initialization
seed_everything(SEED)

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
model = ModelForClassification(model_cfg)

# *** Training the CNN ***

train_dataset = BatchDataset(train_dataset, training_args.batch_size * 100)
# trainer
trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    compute_metrics=lambda p, l: {"acc": accuracy_score(l, p), "f1": f1_score(l, p, average="weighted")},
)

# start mlflow run in experiment
with mlflow.start_run(run.info.run_id):
    # train model
    trainer.train()
    # evaluate model
    print(trainer.evaluate())
