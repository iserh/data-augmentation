from typing import Any, Dict, Optional
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, dataset
from vae import (
    VAEConfig,
    VAEForDataAugmentation,
    augment_data,
    augmentations,
    visualize_latents,
    visualize_real_fake_images,
)
from utils.mlflow import experiments
from sklearn.model_selection import train_test_split
from utils.data import Datasets
import mlflow
from evaluation.models import MLP
from evaluation.trainer import Trainer, TrainingArguments
from dataclasses import fields
from sklearn.decomposition import PCA
import torch
import numpy as np


def train_mlp(
    dataset_name: str,
    dataset_size: int,
    vae_config: Optional[VAEConfig] = None,
    augmentation: str = augmentations.BASELINE,
    alpha: Optional[float] = None,
    alpha2: Optional[float] = None,
    k: Optional[int] = None,
    k2: Optional[int] = None,
) -> None:
    # seeding
    torch.manual_seed(1337)
    np.random.seed(1337)
    mlflow.set_experiment(experiments.MLP)

    # *** data preparation ***

    # evaluation dataset
    test_dataset = Datasets(dataset_name, train=False)
    # the full dataset partition train
    train_dataset_full = Datasets(dataset_name, train=True)
    # extract the image and target tensors
    images, targets = next(iter(DataLoader(train_dataset_full, batch_size=len(train_dataset_full))))

    # split dataset into train and validation parts
    x_train, x_dev, y_train, y_dev = train_test_split(
        images,
        targets,
        stratify=targets,
        shuffle=True,
        train_size=dataset_size / len(train_dataset_full),
        test_size=5000 / len(train_dataset_full),
    )

    # recreate torch datasets
    train_dataset = TensorDataset(x_train, y_train)
    dev_dataset = TensorDataset(x_dev, y_dev)

    # *** Data augmentation ***

    if augmentation != augmentations.BASELINE:
        # load vae model
        vae = VAEForDataAugmentation.from_pretrained(vae_config)
        
        # encode dataset
        original_latents, original_log_vars, original_targets = vae.encode_dataset(train_dataset).tensors
        # augment data
        aug_latents, aug_targets, indices = augment_data(
            original_latents, original_log_vars, original_targets, augmentation, alpha=alpha, alpha2=alpha2, k=k, k2=k2
        )
        # decode augmented latents
        decoded = vae.decode_dataset(TensorDataset(aug_latents, aug_targets))
        # concat datasets
        reals = train_dataset.tensors[0]
        train_dataset = ConcatDataset([train_dataset, decoded])

    # *** MLP model and trainer ***

    print(f"Test: {len(dev_dataset)},\t Train: {len(train_dataset)},\t Dev: {len(test_dataset)}")

    # create mlp model
    mlp = MLP()
    # training arguments for trainer
    training_args = TrainingArguments(epochs=10, save_intervall=None, save_model=False)
    # trainer
    trainer = Trainer(
        args=training_args,
        model=mlp,
        train_dataset=train_dataset,
        test_dataset=dev_dataset,
        eval_dataset=test_dataset,
    )

    # training
    with mlflow.start_run(run_name=augmentation):
        # log params
        if augmentation != augmentations.BASELINE:
            mlflow.log_params({"vae_" + f.name: getattr(vae_config, f.name) for f in fields(vae_config)})
            mlflow.log_params({"alpha": alpha, "alpha2": alpha2, "k": k, "k2": k2, "dataset_size": dataset_size})
            pca = PCA(2).fit(original_latents)
            visualize_latents(
                original_latents, pca, targets=original_targets, color_by_target=True, img_name="original_latents"
            )
            visualize_latents(aug_latents, pca, targets=aug_targets, color_by_target=True, img_name="modified_latents")
            visualize_real_fake_images(
                reals, decoded.tensors[0], n=10, k=k * (k2 if k2 else 1), indices=indices, cols=15
            )

        # train model
        trainer.train()
        # evaluate model
        return trainer.evaluate()


if __name__ == "__main__":
    from utils.mlflow import backend_stores
    mlflow.set_tracking_uri(backend_stores.MNIST)

    train_mlp(
        dataset_name="MNIST",
        dataset_size=10,
        vae_config=VAEConfig(total_epochs=100, epochs=100, z_dim=100, beta=0.3),
        augmentation=[augmentations.EXTRAPOLATION_NOISE, augmentations.FORWARD],
        alpha=[None, 0.5],
        alpha2=0.3,
        k=30,
        k2=None,
    )
