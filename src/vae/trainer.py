"""VAE Trainer."""
from typing import Dict, Optional, Type

import pandas as pd
import torch
from numpy import ceil
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils.trainer import Trainer, TrainingArguments
from utils.visualization import plot_images, plot_points
from vae.models import VAEConfig, VAEForDataAugmentation, VAEOutput
from vae.models.base import VAEModel
from utils import seed_everything


class VAETrainer(Trainer):
    def _compute_metrics(self, outputs: pd.DataFrame, update_best_metric: bool = False) -> Dict[str, float]:
        bce_losses, kl_losses = [*zip(*[(loss.r_loss, loss.kl_loss) for _, loss in outputs["loss"].iteritems()])]
        return {
            "bce_l": sum(bce_losses) / len(bce_losses),
            "kl_l": sum(kl_losses) / len(kl_losses),
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
    seed: Optional[int] = None,
) -> None:
    seed_everything(seed)

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

    latents = vae.encode_dataset(train_dataset).tensors[0][:10_000]
    pca = PCA(2).fit(latents) if latents.size(-1) > 2 else None
    reals, labels = next(iter(DataLoader(train_dataset, batch_size=10_000, num_workers=4)))
    fakes = vae.decode_dataset(TensorDataset(latents, labels)).tensors[0]
    plot_points(
        latents,
        pca=pca,
        labels=labels,
    )
    plot_images(
        images=fakes,
        n=50,
        origins=reals,
        images_title="Fakes",
        origins_title="Original",
        filename="Real-Fake.pdf",
    )

    # random images
    z = torch.FloatTensor(size=(200, latents.size(-1))).uniform_(-1.5, 1.5)
    fakes = vae.decode_dataset(TensorDataset(z, torch.ones((z.size(0),)))).tensors[0]
    plot_points(z, pca=pca, filename="random_latents.pdf")
    plot_images(fakes, 50, filename="random_fakes.pdf")
