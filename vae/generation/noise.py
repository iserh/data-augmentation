"""Data generation by running forward pass through net."""
from typing import Tuple
from vae.commons.visualization import visualize_images
from vae.commons.applied_vae import AppliedVAE
from sklearn.decomposition import PCA

import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data.dataset import TensorDataset
import numpy as np

from utils.data.dataset import LoaderDataset


class Noise:
    def __init__(self, alpha: float, k: int) -> None:
        self.alpha = alpha
        self.k = k

    def __call__(self, z: Tensor, v: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z = z.unsqueeze(1).expand(z.size(0), self.k, *z.size()[1:])
        v = v.unsqueeze(1).expand(v.size(0), self.k, *v.size()[1:])
        y_ = y.unsqueeze(1).expand(y.size(0), self.k, *y.size()[1:])
        normal = torch.normal(0, (0.5 * v).exp())
        z_ = z + self.alpha * normal

        return z_, y_


if __name__ == "__main__":
    import mlflow
    from torch.utils.data import DataLoader, TensorDataset

    from utils.mlflow_utils import Experiment, ExperimentTypes, Roots, get_run, load_pytorch_model
    from vae.commons.applied_vae import AppliedVAE
    from vae.commons.visualization import visualize_latents, visualize_real_fake_images
    from vae.mnist.dataset import MNISTWrapper

    hparams = {
        "EPOCHS": 100,
        "Z_DIM": 2,
        "BETA": 1.0,
    }
    DATASET = "MNIST"
    N_SAMPLES = 1000
    K = 2
    ALPHA = 0.5

    mlflow.set_tracking_uri(Roots.MNIST.value)

    dataset = MNISTWrapper()
    model = load_pytorch_model(get_run(ExperimentTypes.VAETraining, **hparams), chkpt=hparams["EPOCHS"])
    vae = AppliedVAE(model, cuda=True)

    fetch_loader = DataLoader(dataset, batch_size=512, num_workers=4, shuffle=False)
    fetched_set = LoaderDataset(fetch_loader)
    dataloader = DataLoader(fetched_set, batch_size=N_SAMPLES)
    reals, targets = next(iter(dataloader))
    loaded_dataset = TensorDataset(reals, targets)

    encoded_fetch = vae.encode_dataset(loaded_dataset)
    encoded_fetcher = DataLoader(encoded_fetch, batch_size=N_SAMPLES)

    with Experiment(ExperimentTypes.VAEGeneration).new_run("noise") as run:
        mlflow.log_params(hparams)
        mlflow.log_params({"ALPHA": ALPHA, "K": K})
        mlflow.log_metric("N_SAMPLES", N_SAMPLES)
        print("Encoding")
        latents, log_vars, targets = next(iter(encoded_fetcher))
        pca = PCA(2).fit(latents) if latents.size(1) > 2 else None
        visualize_latents(latents, pca, targets, color_by_target=True, img_name="encoded")

        print("Noise")
        noise = Noise(alpha=ALPHA, k=K)
        mod_latents, mod_targets = noise(latents, log_vars, targets)
        mod_latents = mod_latents.reshape(-1, mod_latents.size(-1))
        mod_targets = mod_targets.reshape(-1, mod_targets.size(-1))
        visualize_latents(mod_latents, pca, mod_targets, color_by_target=True, img_name="interpolated")

        fakes_set = vae.decode_dataset(TensorDataset(mod_latents, mod_targets))
        fakes_loader = DataLoader(fakes_set, batch_size=20 * K)

        print("Real - Fake")
        visualize_real_fake_images(reals, next(iter(fakes_loader))[0], n=20, k=K, indices=None)
