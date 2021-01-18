"""Data generation by running forward pass through net."""
import atexit
from typing import Optional

import mlflow
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from utils import get_artifact_path
from vae.pretrained_model import get_pretrained_model

mlflow_run_success = False


class FeatureSpace:
    def __init__(self, epochs: int, z_dim: int, beta: float) -> None:
        self.hparams = {"EPOCHS": epochs, "Z_DIM": z_dim, "BETA": beta}
        self._initialize_mlflow()

    def load_model(self, epoch_chkpt: Optional[int] = None, cuda: bool = True) -> None:
        mlflow.log_param("EPOCH_CHKPT", epoch_chkpt)
        # Use cuda if available
        self.device = "cuda:0" if cuda and torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        self.model = get_pretrained_model(epoch_chkpt if epoch_chkpt is not None else self.hparams["EPOCHS"], **self.hparams).to(self.device)

    def encode(self, dataloader: DataLoader, n_samples: Optional[int] = None) -> torch.Tensor:
        # compute number of batches needed
        n_batches = (n_samples // dataloader.batch_size) if n_samples else len(dataloader)
        mlflow.log_metric("encoded_samples", n_batches * dataloader.batch_size)

        reals, targets, means = [], [], []
        # get means of encoded latent distributions
        with tqdm(total=n_batches) as pbar:
            pbar.set_description("Encoding dataset")
            for _ in range(n_batches):
                x, y = next(iter(dataloader))
                reals.append(x)
                targets.append(y)
                with torch.no_grad():
                    mean, _ = self.model.encoder(x.to(self.device))
                    means.append(mean)

                pbar.update(1)

        self.reals = torch.cat(reals, dim=0)
        self.targets = torch.cat(targets, dim=0)
        self.means = torch.cat(means, dim=0).cpu()

        return self.means

    def visualize_means(self, by_target: bool = True, img_name: str = "means") -> None:
        # reduce dimension to 2d if higher than 2
        if self.means.size(1) > 2:
            pca = PCA(n_components=2).fit(self.means)
            means_2d = pca.transform(self.means)
        else:
            means_2d = self.means

        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
        if by_target:
            unique_targets = torch.unique(self.targets, sorted=True).tolist()
            for t in unique_targets:
                mask = self.targets == t
                ax.scatter(*means_2d[mask].T, label=f"{t}")
        else:
            ax.scatter(*means_2d.T, label="means")
        plt.legend()
        plt.savefig(self.artifact_path / (img_name + ".png"))
        plt.close()

    # *** private functions ***

    def _initialize_mlflow(self) -> None:
        # initialize mlflow experiment
        experiment_name = "Feature Space"
        run_name = "2D"
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            self.experiment = mlflow.get_experiment(mlflow.create_experiment(experiment_name))
        # start new run
        self.run = mlflow.start_run(experiment_id=self.experiment.experiment_id, run_name=run_name)
        self.artifact_path = get_artifact_path(self.run)


# exit handling
def exit_hook():
    global mlflow_run_success
    if mlflow_run_success:
        mlflow.end_run()
    else:
        mlflow.end_run("KILLED")


if __name__ == "__main__":
    atexit.register(exit_hook)
    from utils.config import mlflow_roots
    from vae.setup import get_dataloader

    DATASET = "MNIST"
    EPOCHS = 100
    EPOCH_CHKPT = 100
    Z_DIM = 2
    BETA = 1.0
    N_SAMPLES = None

    mlflow.set_tracking_uri(mlflow_roots[DATASET])
    dataloader = get_dataloader(DATASET, train=True, shuffle=True, batch_size=512)

    gen = FeatureSpace(epochs=EPOCHS, z_dim=Z_DIM, beta=BETA)
    gen.load_model(EPOCH_CHKPT, cuda=True)
    gen.encode(dataloader, N_SAMPLES)
    gen.visualize_means(by_target=True)

    mlflow_run_success = True
