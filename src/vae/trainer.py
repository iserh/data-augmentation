"""VAE Trainer."""
from typing import Dict, Optional, Type

import pandas as pd
import torch
from numpy import ceil
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils.trainer import Trainer, TrainingArguments
from vae.models import VAEConfig, VAEForDataAugmentation, VAEOutput
from vae.models.base import VAEModel
from vae.visualization import visualize_images, visualize_latents


class VAETrainer(Trainer):
    def log_epoch(self, outputs: pd.DataFrame, validate: bool = False) -> Dict[str, float]:
        metrics = self.epoch_metrics(outputs["prediction"], outputs["label"]) if self.epoch_metrics else {}
        bce_losses, kl_losses = [*zip(*[(loss.r_loss, loss.kl_loss) for _, loss in outputs["loss"].iteritems()])]
        return {
            "bce_l": sum(bce_losses) / len(bce_losses),
            "kl_l": sum(kl_losses) / len(kl_losses),
            **metrics,
        }

    def log_step(self, outputs: pd.DataFrame, validate: bool = False) -> Dict[str, float]:
        metrics = self.step_metrics(outputs["prediction"], outputs["label"]) if self.step_metrics else {}
        bce_losses, kl_losses = [*zip(*[(loss.r_loss, loss.kl_loss) for _, loss in outputs["loss"].iteritems()])]
        return {
            "bce_l": sum(bce_losses) / len(bce_losses),
            "kl_l": sum(kl_losses) / len(kl_losses),
            **metrics,
        }

    def train_step(self, x: Tensor, y: Optional[Tensor] = None) -> Dict[str, float]:
        # forward pass
        output: VAEOutput = self.model(x, y)
        # update parameters
        self.optim.zero_grad()
        output.loss.backward()
        self.optim.step()
        # return losses
        return {"loss": output.loss.item(), "label": y.cpu()}

    @torch.no_grad()
    def test_step(self, x: Tensor, y: Optional[Tensor] = None) -> Dict[str, float]:
        # forward pass
        output: VAEOutput = self.model(x, y)
        # return losses
        return {"loss": output.loss.item(), "label": y.cpu()}


def train_vae(
    training_args: TrainingArguments,
    train_dataset: Dataset,
    model_architecture: Type[VAEModel],
    vae_config: VAEConfig,
    save_every_n_epochs: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    # seed
    if seed is not None:
        torch.manual_seed(seed)
        training_args.seed = seed

    if save_every_n_epochs is not None:
        training_args.save_intervall = save_every_n_epochs * ceil(len(train_dataset) / training_args.batch_size).astype(
            int
        )

    # create model
    model = model_architecture(vae_config)

    # trainer
    trainer = VAETrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
    )

    # start training
    model = trainer.train()
    del model

    # visualization
    vae = VAEForDataAugmentation.from_pretrained(vae_config, epochs=training_args.epochs)

    encoded = vae.encode_dataset(train_dataset)
    pca = PCA(2).fit(encoded.tensors[0]) if encoded.tensors[0].size(1) > 2 else None
    reals, labels = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset), num_workers=4)))
    fakes = vae.decode_dataset(TensorDataset(encoded.tensors[0], encoded.tensors[1])).tensors[0]
    visualize_latents(
        encoded.tensors[0],
        pca=pca,
        labels=labels,
        color_by_label=True,
    )
    visualize_images(
        images=fakes,
        n=50,
        heritages=reals,
        cols=5,
        img_title="Fakes",
        heritage_title="Original",
        filename="Real-Fake.png",
    )

    # random images
    z = torch.normal(0, 1, size=(200, vae_config.z_dim))
    labels = torch.ones((200,))  # arbitrary labels
    fakes = vae.decode_dataset(TensorDataset(z, labels)).tensors[0]
    visualize_latents(z, pca=pca, filename="random_latents.png")
    visualize_images(fakes, 50, cols=5, filename="random_fakes.png")


if __name__ == "__main__":
    import mlflow
    import torch

    import vae
    from utils.data import load_splitted_datasets, load_unsplitted_dataset
    from utils.mlflow import backend_stores
    from utils.trainer import TrainingArguments
    from vae.models import VAEConfig
    from vae.models.architectures import VAEModelV1 as VAEModelVersion
    from utils.data import BatchDataset

    vae.models.base.model_store = "pretrained_models/MNIST"

    # *** Seeding, loading data & setting up mlflow logging ***

    DATASET = "MNIST"
    SEED = 1337

    # set the backend store uri of mlflow
    mlflow.set_tracking_uri(getattr(backend_stores, DATASET))
    # seed torch
    torch.manual_seed(SEED)

    # *** VAE Parameters ***

    MULTI_VAE = False
    VAE_EPOCHS = 25
    Z_DIM = 3
    BETA = 1.0

    # *** Training the VAE ***

    # set mlflow experiment
    mlflow.set_experiment(f"Z_DIM {Z_DIM}")
    vae.models.base.model_store = f"pretrained_models/MNIST/NO_OTHER/{'MULTI' if MULTI_VAE else 'SINGLE'}/Z_DIM {Z_DIM}"

    if MULTI_VAE:
        datasets, dataset_info = load_splitted_datasets(DATASET, others=False)
        for label, train_dataset, class_count in zip(dataset_info["classes"], datasets, dataset_info["class_counts"]):
            with mlflow.start_run():
                mlflow.log_param("class_count", class_count)
                train_vae(
                    TrainingArguments(VAE_EPOCHS, batch_size=128),
                    train_dataset=BatchDataset(train_dataset, 100 * 128),
                    model_architecture=VAEModelVersion,
                    vae_config=VAEConfig(z_dim=Z_DIM, beta=BETA, attr={"label": label}),
                    save_every_n_epochs=5,
                    seed=SEED,
                )
    else:
        train_dataset, _ = load_unsplitted_dataset(DATASET)
        with mlflow.start_run():
            train_vae(
                TrainingArguments(VAE_EPOCHS, batch_size=128),
                train_dataset=train_dataset,
                model_architecture=VAEModelVersion,
                vae_config=VAEConfig(z_dim=Z_DIM, beta=BETA),
                save_every_n_epochs=50,
                seed=SEED,
            )
