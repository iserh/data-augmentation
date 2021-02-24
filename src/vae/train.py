"""Train VAE example."""
from typing import Optional, Type

import mlflow
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils.trainer import TrainingArguments
from vae.models import VAEConfig, VAEForDataAugmentation
from vae.models.base import VAEModel
from vae.trainer import VAETrainer
from vae.visualization import visualize_images, visualize_latents, visualize_real_fake_images
from numpy import ceil


def train_vae_on_classes(
    training_args: TrainingArguments,
    train_dataset: Dataset,
    test_dataset: Dataset,
    vae_config: VAEConfig,
    model_architecture: Type[VAEModel],
    save_every_n_epochs: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    # load train/test data
    x_train, y_train = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))
    x_test, y_test = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset))))

    unique_labels = torch.unique(y_train, sorted=True)

    for label in unique_labels:
        # seed
        if seed is not None:
            torch.manual_seed(seed)
        # create model
        vae_config.attr = {"label": label.item()}
        model = model_architecture(vae_config)

        # get data of label
        train_mask = y_train == label
        test_mask = y_test == label
        train_dataset = TensorDataset(x_train[train_mask], y_train[train_mask])
        test_dataset = TensorDataset(x_test[test_mask], y_test[test_mask])

        if save_every_n_epochs is not None:
            training_args.save_intervall = save_every_n_epochs * ceil(len(train_dataset) / training_args.batch_size).astype(int)

        # trainer
        trainer = VAETrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            dev_dataset=test_dataset,
        )
        print(f"Training VAE on label {label.item()}")
        # start training
        with mlflow.start_run():
            model = trainer.train()

            # visualization
            vae = VAEForDataAugmentation.from_pretrained(vae_config, epochs=training_args.epochs)

            encoded = vae.encode_dataset(test_dataset)
            fakes = vae.decode_dataset(TensorDataset(encoded.tensors[0], encoded.tensors[1])).tensors[0]
            pca = PCA(2).fit(encoded.tensors[0]) if vae_config.z_dim > 2 else None
            visualize_latents(
                encoded.tensors[0],
                pca=pca,
                targets=test_dataset.tensors[1],
                color_by_target=True,
            )
            visualize_real_fake_images(test_dataset.tensors[0], fakes, n=50, cols=8)

            # random images
            z = torch.normal(0, 1, size=(200, vae_config.z_dim))
            labels = torch.ones((200,)) * label
            fakes = vae.decode_dataset(TensorDataset(z, labels)).tensors[0]
            visualize_latents(z, pca=pca, img_name="random_latents")
            visualize_images(fakes, 50, "random_fakes", cols=8)


def train_vae_on_dataset(
    training_args: TrainingArguments,
    train_dataset: Dataset,
    test_dataset: Dataset,
    vae_config: VAEConfig,
    model_architecture: Type[VAEModel],
    save_every_n_epochs: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    # seed
    if seed is not None:
        torch.manual_seed(seed)
    # create model
    model = model_architecture(vae_config)
    if save_every_n_epochs is not None:
        training_args.save_intervall = save_every_n_epochs * ceil(len(train_dataset) / training_args.batch_size).astype(int)

    # trainer
    trainer = VAETrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        dev_dataset=test_dataset,
    )
    

    print("Training VAE on dataset")
    # start training
    with mlflow.start_run():
        model = trainer.train()

        # visualization
        vae = VAEForDataAugmentation.from_pretrained(vae_config, epochs=training_args.epochs)

        encoded = vae.encode_dataset(test_dataset)
        reals, labels = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=4)))
        fakes = vae.decode_dataset(TensorDataset(encoded.tensors[0], encoded.tensors[1])).tensors[0]
        visualize_latents(
            encoded.tensors[0],
            pca=PCA(2).fit(encoded.tensors[0]),
            targets=labels,
            color_by_target=True,
        )
        visualize_real_fake_images(reals, fakes, n=50, cols=8)

        # random images
        z = torch.normal(0, 1, size=(200, vae_config.z_dim))
        labels = torch.ones((200,))  # arbitrary labels
        fakes = vae.decode_dataset(TensorDataset(z, labels)).tensors[0]
        visualize_latents(z, pca=PCA(2).fit(encoded.tensors[0]), img_name="random_latents")
        visualize_images(fakes, 50, "random_fakes", cols=8)


if __name__ == "__main__":
    from utils.mlflow import backend_stores
    from utils.trainer import TrainingArguments
    from vae.models import VAEConfig
    from vae.models.architectures import VAEModelV1

    from utils.data import load_datasets

    # *** Seeding, loading data & setting up mlflow logging ***

    DATASET = "MNIST"
    DATASET_LIMIT = 50
    SEED = 1337

    # set the backend store uri of mlflow
    mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
    # seed torch
    torch.manual_seed(SEED)
    # load datasets
    train_dataset, vae_train_dataset, val_dataset, test_dataset = load_datasets(DATASET)

    # *** Training the VAE ***

    for z_dim in [3]:
        for beta in [1.0]:
            for vae_epochs in [200]:
                # set mlflow experiment
                mlflow.set_experiment(f"Z_DIM {z_dim}")
                print(f"Training VAE with {z_dim=}, {beta=}, {vae_epochs=}")
                # create a vae config
                vae_config = VAEConfig(z_dim=z_dim, beta=beta)
                # train vae
                train_vae_on_classes(
                    training_args=TrainingArguments(vae_epochs, seed=SEED, batch_size=64),
                    train_dataset=vae_train_dataset,
                    test_dataset=val_dataset,
                    vae_config=vae_config,
                    model_architecture=VAEModelV1,
                    save_every_n_epochs=25,
                    seed=SEED,
                )
