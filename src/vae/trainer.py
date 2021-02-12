"""VAE Trainer."""
from typing import Dict

import mlflow
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.trainer import Trainer


class VAETrainer(Trainer):
    def train_epoch(self, epoch: int, train_loader: DataLoader, pbar: tqdm) -> Dict[str, float]:
        self.model.train()
        running_bce_l, running_kl_l = 0, 0
        for step, (x_true, _) in enumerate(train_loader, start=1):
            # train step, update running loss
            bce_l, kl_l = self.train_step(x_true.to(self.device, non_blocking=True))["loss"]
            running_bce_l += bce_l
            running_kl_l += kl_l
            # log to mlflow
            mlflow.log_metric("epoch", epoch - 1 + (0.5 * step / len(train_loader)))
            # progress bar
            pbar.set_postfix({"bce_l": running_bce_l / step, "kl_l": running_kl_l / step})
            pbar.update(1)
        # return runnning loss
        return {
            "train_bce_l": running_bce_l / len(train_loader),
            "train_kl_l": running_kl_l / len(train_loader),
        }

    def test_epoch(self, epoch: int, test_loader: DataLoader, pbar: tqdm) -> Dict[str, float]:
        self.model.eval()
        running_bce_l, running_kl_l = 0, 0
        with tqdm(total=len(test_loader), desc=f"Test epoch {epoch}/{self.args.epochs}", leave=False) as test_pbar:
            for step, (x_true, _) in enumerate(test_loader, start=1):
                # test step, update running loss
                bce_l, kl_l = self.test_step(x_true.to(self.device, non_blocking=True))["loss"]
                running_bce_l += bce_l
                running_kl_l += kl_l
                # log to mlflow
                mlflow.log_metric("epoch", epoch - 0.5 + (0.5 * step / len(test_loader)))
                # progress bar
                test_pbar.set_postfix({"bce_l": running_bce_l / step, "kl_l": running_kl_l / step})
                test_pbar.update(1)
        # return running loss
        return {
            "test_bce_l": running_bce_l / len(test_loader),
            "test_kl_l": running_kl_l / len(test_loader),
        }
