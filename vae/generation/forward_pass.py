"""Data generation by running forward pass through net."""
from typing import Optional, Tuple

import mlflow
import numpy as np
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from utils import get_artifact_path
from vae.pretrained_model import get_pretrained_model

mlflow_run_success = False


class VAEGenerator:
    def __init__(self, epochs: int, z_dim: int, beta: float) -> None:
        self.hparams = {"EPOCHS": epochs, "Z_DIM": z_dim, "BETA": beta}
        self._initialize_mlflow()

    def load_model(self, epoch_chkpt: Optional[int] = None, cuda: bool = True) -> None:
        mlflow.log_param("EPOCH_CHKPT", epoch_chkpt)
        # Use cuda if available
        self.device = "cuda:0" if cuda and torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        self.model = get_pretrained_model(
            epoch_chkpt if epoch_chkpt is not None else self.hparams["EPOCHS"], **self.hparams
        ).to(self.device)

    def augment_single_example(self, dataloader: DataLoader, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mlflow.log_metric("single_augment_samples", n_samples)

        real_x, real_y = next(iter(dataloader))
        self.reals = torch.cat([real_x[:1]] * n_samples, dim=0)
        self.targets = torch.cat([real_y[:1]] * n_samples, dim=0)

        fakes, z_samples, means = [], [], []

        with tqdm(total=n_samples) as pbar:
            pbar.set_description("Augmenting single example")
            for _ in range(n_samples):
                with torch.no_grad():
                    fake, mean, _, z = self.model(real_x[:1].to(self.device))
                    means.append(mean)
                    z_samples.append(z)
                    fakes.append(fake)

                pbar.update(1)

        self.fakes = torch.cat(fakes, dim=0).cpu()
        self.z_samples = torch.cat(z_samples, dim=0).cpu()
        self.means = torch.cat(means, dim=0).cpu()

    def generate_new_data(
        self, dataloader: DataLoader, n_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute number of batches needed
        n_batches = np.ceil(n_samples / dataloader.batch_size).astype(int) if n_samples else len(dataloader)
        mlflow.log_metric("generated_samples", n_batches * dataloader.batch_size)

        reals, fakes, targets = [], [], []
        z_samples, means = [], []
        # get means of encoded latent distributions
        with tqdm(total=n_batches) as pbar:
            pbar.set_description("Generating new data")
            for _ in range(n_batches):
                x, y = next(iter(dataloader))
                reals.append(x)
                targets.append(y)
                with torch.no_grad():
                    fake, mean, _, z = self.model(x.to(self.device))
                    means.append(mean)
                    z_samples.append(z)
                    fakes.append(fake)

                pbar.update(1)

        self.reals = torch.cat(reals, dim=0)
        self.fakes = torch.cat(fakes, dim=0).cpu()
        self.targets = torch.cat(targets, dim=0)
        self.z_samples = torch.cat(z_samples, dim=0).cpu()
        self.means = torch.cat(means, dim=0).cpu()

        return self.fakes, self.targets

    def visualize_z_samples(self, by_target: bool = True, img_name: str = "generated_z") -> None:
        # reduce dimension to 2d if higher than 2
        if self.means.size(1) > 2:
            pca = PCA(n_components=2).fit(self.means)
            z_samples_2d = pca.transform(self.z_samples)
        else:
            z_samples_2d = self.z_samples

        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(xlim=(-4, 4), ylim=(-4, 4))
        if by_target:
            unique_targets = torch.unique(self.targets, sorted=True).tolist()
            for t in unique_targets:
                mask = self.targets == t
                ax.scatter(*z_samples_2d[mask].T, label=f"{t}")
        else:
            ax.scatter(*z_samples_2d.T, label="z_samples")
        plt.legend()
        plt.savefig(self.artifact_path / (img_name + ".png"))
        plt.close()

    def visualize_real_fake_images(self, n: int, img_name: str = "real_fake") -> None:
        # Plot the real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    self.reals[:n],
                    padding=5,
                    normalize=True,
                ),
                (1, 2, 0),
            )
        )

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    self.fakes[:n],
                    padding=5,
                    normalize=True,
                ),
                (1, 2, 0),
            )
        )
        plt.savefig(self.artifact_path / (img_name + ".png"))

    # *** private functions ***

    def _initialize_mlflow(self) -> None:
        # initialize mlflow experiment
        experiment_name = "VAE Generation"
        run_name = "forward pass"
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            self.experiment = mlflow.get_experiment(mlflow.create_experiment(experiment_name))
        # start new run
        self.run = mlflow.start_run(experiment_id=self.experiment.experiment_id, run_name=run_name)
        self.artifact_path = get_artifact_path(self.run)
        mlflow.log_params(self.hparams)


# exit handling
def exit_hook():
    global mlflow_run_success
    if mlflow_run_success:
        mlflow.end_run()
    else:
        mlflow.end_run("KILLED")


if __name__ == "__main__":
    import atexit

    atexit.register(exit_hook)
    from utils.config import mlflow_roots
    from vae.setup import get_dataloader

    mlflow.set_tracking_uri("./experiments/CelebA")

    DATASET = "MNIST"
    EPOCHS = 100
    EPOCH_CHKPT = 0
    Z_DIM = 2
    BETA = 1.0
    N_SAMPLES = 1024

    mlflow.set_tracking_uri(mlflow_roots[DATASET])
    dataloader = get_dataloader(DATASET, train=False, shuffle=True)

    gen = VAEGenerator(epochs=EPOCHS, z_dim=Z_DIM, beta=BETA)
    gen.load_model(epoch_chkpt=EPOCH_CHKPT, cuda=True)
    gen.generate_new_data(dataloader, N_SAMPLES)
    gen.visualize_z_samples(by_target=True, img_name="generated_z")
    gen.visualize_real_fake_images(64, img_name="real_fake")
    gen.augment_single_example(dataloader, N_SAMPLES)
    gen.visualize_z_samples(by_target=True, img_name="generated_z_single")
    gen.visualize_real_fake_images(64, img_name="real_fake_single")

    mlflow_run_success = True
