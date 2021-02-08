from typing import Any, Dict, Optional, Tuple
from utils.integrations.mlflow_integration import ExperimentName
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from vae import (
    VAEModel,
    VAEConfig,
    VAEForDataAugmentation,
    Interpolation,
    Noise,
    visualize_latents,
    visualize_real_fake_images,
)
from sklearn.model_selection import train_test_split
from utils.data import Datasets
import mlflow
from evaluation.models import MLP
from evaluation.trainer import Trainer, TrainingArguments
from enum import Enum
from dataclasses import fields
from sklearn.decomposition import PCA
from torch import Tensor
import torch
import numpy as np
import optuna


class Augmentations(Enum):
    interpolation: str = "interpolation"
    extrapolation: str = "extrapolation"
    noise: str = "noise"
    noise_extra: str = "noise_extra"
    noise_inter: str = "noise_inter"
    vae: str = "vae"


def augment_data(
    original_latents: Tensor,
    original_targets: Tensor,
    augmentation: Augmentations,
    **kwargs: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Tensor]:
    if augmentation == Augmentations.interpolation:
        # augment using interpolation
        interpolation = Interpolation(alpha=kwargs["alpha"], k=kwargs["k"], return_indices=True)
        latents, targets, indices = interpolation(original_latents, original_targets)
    elif augmentation == Augmentations.extrapolation:
        # augment using extrapolation
        interpolation = Interpolation(alpha=-kwargs["alpha"], k=kwargs["k"], return_indices=True)
        latents, targets, indices = interpolation(original_latents, original_targets)
    elif augmentation == Augmentations.noise:
        # augment using noise
        noise = Noise(alpha=kwargs["alpha"], k=kwargs["k"], std=original_latents.std(), return_indices=True)
        latents, targets, indices = noise(original_latents, original_targets)
    elif augmentation == Augmentations.noise_extra:
        # augment using interpolation and noise
        interpolation = Interpolation(alpha=kwargs["alpha"], k=kwargs["k"], return_indices=True)
        latents, targets, indices = interpolation(original_latents, original_targets)
        noise = Noise(
            alpha=kwargs["alpha2"], k=kwargs["k2"], std=latents.std(), return_indices=True, indices_before=indices
        )
        latents, targets, indices = noise(latents, targets)
    elif augmentation == Augmentations.noise_inter:
        # augment using interpolation and noise
        interpolation = Interpolation(alpha=-kwargs["alpha"], k=kwargs["k"], return_indices=True)
        latents, targets, indices = interpolation(original_latents, original_targets)
        noise = Noise(
            alpha=kwargs["alpha2"], k=kwargs["k2"], std=latents.std(), return_indices=True, indices_before=indices
        )
        latents, targets, indices = noise(latents, targets)
    else:
        raise NotImplementedError("Augmentation method not implemented!")

    return latents, targets, indices


def train_mlp(
    dataset_name: str,
    dataset_size: int,
    vae_config: Optional[VAEConfig] = None,
    augmentation: Optional[Augmentations] = None,
    alpha: Optional[float] = None,
    alpha2: Optional[float] = None,
    k: Optional[int] = None,
    k2: Optional[int] = None,
) -> None:
    # seeding
    torch.manual_seed(1337)
    np.random.seed(1337)
    mlflow.set_experiment(ExperimentName.MLP.value)

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

    if augmentation is not None:
        # load vae model
        vae = VAEForDataAugmentation.from_pretrained(vae_config)
        if augmentation == Augmentations.vae:
            # batched forward pass through model
            x_train = x_train.unsqueeze(0).expand(k, *x_train.size())
            y_train = y_train.unsqueeze(0).expand(k, *y_train.size())
            train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=512)
            decoded = torch.cat([vae(x.to(vae.device)).reconstructed for x, _ in train_loader], dim=0)
        else:
            # encode dataset
            original_latents, original_targets = vae.encode_dataset(train_dataset).tensors
            # augment data
            aug_latents, aug_targets, indices = augment_data(
                original_latents, original_targets, augmentation, alpha=alpha, alpha2=alpha2, k=k, k2=k2
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
    with mlflow.start_run(run_name=augmentation.value if augmentation else "baseline"):
        # log params
        if augmentation is not None and augmentation != Augmentations.vae:
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


def objective(trial: optuna.Trial) -> float:
    DATASET = "MNIST"
    DATASET_SIZE = 50
    VAE_EPOCHS = trial.suggest_int("VAE_EPOCHS", 0, 100)
    VAE_Z_DIM = trial.suggest_int("VAE_Z_DIM", 0, 100)
    VAE_BETA = trial.suggest_float("VAE_BETA", 0, 1.0)
    AUGMENTATION = Augmentations[trial.suggest_categorical("AUGMENTATION", [aug.value for aug in list(Augmentations)])]
    ALPHA = trial.suggest_float("ALPHA", 0, 1)
    ALPHA2 = trial.suggest_float("ALPHA2", 0, 1)
    K = trial.suggest_int("K", 0, 100)
    K2 = trial.suggest_int("K2", 0, 100)

    eval_scores = train_mlp(
        dataset_name=DATASET,
        dataset_size=DATASET_SIZE,
        vae_config=VAEConfig(100, VAE_EPOCHS, VAE_Z_DIM, VAE_BETA),
        augmentation=AUGMENTATION,
        alpha=ALPHA,
        alpha2=ALPHA2,
        k=K,
        k2=K2,
    )

    return eval_scores["eval_acc"]


if __name__ == "__main__":
    from utils.integrations import BackendStore
    mlflow.set_tracking_uri(BackendStore.MNIST.value)
    search_space = {
        "VAE_EPOCHS": [100],
        "VAE_Z_DIM": [50, 100],
        "VAE_BETA": [1.0],
        "AUGMENTATION": ["vae"],
        "ALPHA": [1],
        "ALPHA2": [1],
        "K": range(1, 5, 1),
        "K2": [1],
    }
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(objective, n_trials=2 * 5)
