"""VAE Trainer."""
from typing import Dict

import pandas as pd

from utils.trainer import Trainer


class VAETrainer(Trainer):
    def log_epoch(self, outputs: pd.DataFrame) -> Dict[str, float]:
        metrics = (
            self.epoch_metrics(outputs["prediction"].tolist(), outputs["label"].tolist()) if self.epoch_metrics else {}
        )
        bce_losses, kl_losses = [*zip(*outputs["loss"].tolist())]
        return {
            "bce_l": sum(bce_losses) / len(bce_losses),
            "kl_l": sum(kl_losses) / len(kl_losses),
            **metrics,
        }

    def log_step(self, outputs: pd.DataFrame) -> Dict[str, float]:
        metrics = (
            self.step_metrics(outputs.iloc[-1]["prediction"], outputs.iloc[-1]["label"]) if self.step_metrics else {}
        )
        return {
            "bce_l": outputs["loss"].tolist()[-1][0],
            "kl_l": outputs["loss"].tolist()[-1][1],
            **metrics,
        }
