"""VAE Trainer."""
from typing import Dict, Optional

import pandas as pd

from utils.trainer import Trainer
from torch import Tensor
import torch

from vae.models.base import VAEOutput


class VAETrainer(Trainer):
    def log_epoch(self, outputs: pd.DataFrame, train: bool = False) -> Dict[str, float]:
        metrics = (
            self.epoch_metrics(outputs["prediction"], outputs["label"]) if self.epoch_metrics else {}
        )
        bce_losses, kl_losses = [*zip(*[(loss.r_loss, loss.kl_loss) for _, loss in outputs["loss"].iteritems()])]
        return {
            "bce_l": sum(bce_losses) / len(bce_losses),
            "kl_l": sum(kl_losses) / len(kl_losses),
            **metrics,
        }

    def log_step(self, outputs: pd.DataFrame, train: bool = False) -> Dict[str, float]:
        metrics = (
            self.step_metrics(outputs["prediction"], outputs["label"]) if self.step_metrics else {}
        )
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
