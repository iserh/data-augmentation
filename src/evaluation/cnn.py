from dataclasses import fields
from typing import Dict, Optional, Union

import mlflow
import torch
from sklearn.decomposition import PCA
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, Dataset
import torch.nn as nn

from torch import Tensor

from vae.models import VAEForDataAugmentation, VAEConfig
from vae.generation import augment_data
from vae.visualization import visualize_latents, visualize_real_fake_images
from utils.trainer import Trainer, TrainingArguments
from utils.models import BaseModel, ModelOutput, ModelConfig
from sklearn.metrics import accuracy_score, f1_score


class CNNModel(BaseModel):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super(CNNModel, self).__init__(config)
        self.sequential = nn.Sequential(
            # input size: (nc) x 28 x 28
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (64) x 28 x 28
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (128) x 28 x 28
            nn.MaxPool2d(2),
            # state size: (128) x 14 x 14
            nn.Dropout(0.5),
            # ---------------------------
            # state size: (128) x 14 x 14
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (128) x 14 x 14
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            # state size: (128) x 14 x 14
            nn.MaxPool2d(2),
            # state size: (128) x 7 x 7
            nn.Dropout(0.5),
            # ---------------------------
            # state size: (128) x 7 x 7
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # state size: 128
            nn.Linear(128, 10),
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: Tensor, y: Tensor) -> ModelOutput:
        pred: Tensor = self.sequential(x)
        loss = self.criterion(pred, y)
        return ModelOutput(loss=loss, prediction=pred.argmax(1))


def perform_data_augmentation(
    dataset: Dataset,
    vae_config: VAEConfig,
    vae_epochs: int,
    augmentation_method: str,
    augmentation_params: Dict[str, Union[float, int]],
    seed: Optional[int] = None,
) -> Dataset:
    n_img_plots = 10
    # seeding
    if seed is not None:
        torch.manual_seed(seed)

    # log vae config
    mlflow.log_params({"vae_" + f.name: getattr(vae_config, f.name) for f in fields(vae_config)})
    # log augmentation parameters
    mlflow.log_params(augmentation_params)

    # the real images in the dataset
    reals = next(iter(DataLoader(dataset, batch_size=n_img_plots)))[0]

    # *** Data augmentation ***

    # load vae model
    vae = VAEForDataAugmentation.from_pretrained(vae_config, epochs=vae_epochs)
    # encode dataset
    original_latents, original_log_vars, original_targets = vae.encode_dataset(dataset).tensors
    # augment data - get augmented latents, targets and the indices used for augmentation (for visualization)
    aug_latents, aug_targets, aug_indices = augment_data(
        original_latents, original_log_vars, original_targets, augmentation_method, **augmentation_params
    )
    # decode augmented latents
    decoded = vae.decode_dataset(TensorDataset(aug_latents, aug_targets))

    # *** Visualization ***

    # pca for 2d view
    pca = PCA(2).fit(original_latents)
    # visualize encoded latents
    visualize_latents(
        original_latents, pca, targets=original_targets, color_by_target=True, img_name="original_latents"
    )
    # visualize augmented latents
    visualize_latents(aug_latents, pca, targets=aug_targets, color_by_target=True, img_name="augmented_latents")
    # visualize real - fake comparison
    visualize_real_fake_images(
        reals, decoded.tensors[0][n_img_plots], n=n_img_plots, k=augmentation_params["k"] * augmentation_params.get("k2", 1), indices=aug_indices, cols=15
    )

    # *** Return ***

    # concat the original and the augmented dataset
    concat_dataset = ConcatDataset([dataset, decoded])
    print(f"Augmented dataset from {'X'} samples per class ({len(dataset)} total) to {'X'} samples per class ({len(concat_dataset)} total)")
    return concat_dataset


def train_cnn(training_args: TrainingArguments, train_dataset: Dataset, dev_dataset: Dataset, test_dataset: Dataset):
    # create mlp model
    mlp = CNNModel()
    # trainer
    trainer = Trainer(
        args=training_args,
        model=mlp,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        step_metrics=step_metrics,
    )
    # train model
    trainer.train()
    # evaluate model
    return trainer.evaluate()


def step_metrics(predictions: Tensor, labels: Tensor) -> Dict[str, float]:
    return {
        "acc": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }


if __name__ == "__main__":
    from utils.mlflow import backend_stores
    from utils.data import get_dataset
    from torch.utils.data import random_split
    from vae.generation import augmentations

    DATASET = "MNIST"
    DATASET_LIMIT = 100
    AUGMENTATION = augmentations.FORWARD
    VAE_EPOCHS = 20
    training_args = TrainingArguments(epochs=10, save_intervall=None, save_model=False)
    vae_config = VAEConfig(z_dim=10, beta=1.0)
    augmentation_params = {"k": 3}
    mlflow.set_tracking_uri(getattr(backend_stores, DATASET))

    # load test dataset
    test_dataset = get_dataset(DATASET, train=False)
    # load train dataset
    train_dataset = get_dataset(DATASET, train=True)
    # limit train dataset corresponding to DATASET_LIMIT and add 5000 for dev dataset
    train_dataset, dev_dataset, _ = random_split(train_dataset, [DATASET_LIMIT, 5000, len(train_dataset) - (DATASET_LIMIT + 5000)])

    # * VAE Training: uncomment these 2 line to train the vae
    # from vae import train_vae_on_dataset
    # train_vae_on_dataset(TrainingArguments(epochs=VAE_EPOCHS, seed=1337), train_dataset, test_dataset, vae_config

    mlflow.set_experiment("CNN Training")
    with mlflow.start_run(run_name=AUGMENTATION or "baseline"):
        # perform data augmentation
        if AUGMENTATION is not None:
            train_dataset = perform_data_augmentation(train_dataset, vae_config, VAE_EPOCHS, AUGMENTATION, augmentation_params, seed=1337)
            dev_dataset = perform_data_augmentation(train_dataset, vae_config, VAE_EPOCHS, AUGMENTATION, augmentation_params, seed=1337)

        # train cnn
        results = train_cnn(
            training_args=training_args,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
        )

        print(results)
