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
from utils.visualization import plot_images, plot_points


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
    plot_points(
        encoded.tensors[0],
        pca=pca,
        labels=labels,
    )
    plot_images(
        images=fakes,
        n=50,
        origins=reals,
        images_title="Fakes",
        origins_title="Original",
        filename="Real-Fake.png",
    )

    # random images
    z = torch.normal(0, 1, size=(200, vae_config.z_dim))
    labels = torch.ones((200,))  # arbitrary labels
    fakes = vae.decode_dataset(TensorDataset(z, labels)).tensors[0]
    plot_points(z, pca=pca, filename="random_latents.png")
    plot_images(fakes, 50, filename="random_fakes.png")
