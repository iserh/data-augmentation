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


class Interpolation:
    def __init__(self, alpha: float, k: int = 3, return_indices: bool = False) -> None:
        self.alpha = alpha
        self.k = k
        self.return_indices = return_indices

    def __call__(self, z: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # build nearest neighbour tree
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm="ball_tree").fit(z)
        # get indices of k nearest neighbours for each latent vector
        _, indices = nbrs.kneighbors(z)
        # generate k new latents for each original latent vector
        # by interpolating between the k'th nearest neighbour
        z_ = torch.empty((z.size(0), self.k, *z.size()[1:]), device=z.device)
        y_ = torch.empty((y.size(0), self.k, *y.size()[1:]), device=y.device)
        for i in range(z.size(0)):
            # each latent vector generates 'n_neighbor' new latent vectors
            for j, k in enumerate(indices[i]):
                # interpolate between latent vector and the k'th nearest neighbour
                z_[i, j] = (z[k] - z[i]) * self.alpha + z[i]
                # save the target too
                y_[i, j] = y[i]
        # return new modified latents and the corresponding targets as tensors
        return (z_, y_, indices) if self.return_indices else (z_, y_)


def interpolate_along_dimension(latents: Tensor, targets: Tensor, n_steps: int) -> Tensor:
    unique_targets = torch.unique(targets, sorted=True).tolist()
    interpolated, corresponding_targets = [], []
    for i in unique_targets:
        mask = targets == i
        pca = PCA(1).fit(latents[mask.flatten()])
        pc = pca.transform(latents)
        x = np.expand_dims(np.linspace(pc.min(), pc.max(), n_steps), axis=1)
        interpolated.append(torch.Tensor(pca.inverse_transform(x)))
        corresponding_targets.append(torch.Tensor([i] * n_steps))

    return torch.cat(interpolated, dim=0), torch.cat(corresponding_targets, dim=0)



if __name__ == "__main__":
    import mlflow
    from torch.utils.data import DataLoader, TensorDataset

    from utils.mlflow_utils import Experiment, ExperimentTypes, Roots, get_run, load_pytorch_model
    from vae.commons.applied_vae import AppliedVAE
    from vae.commons.visualization import visualize_latents, visualize_real_fake_images
    from vae.mnist.dataset import MNISTWrapper

    hparams = {
        "EPOCHS": 100,
        "Z_DIM": 20,
        "BETA": 1.0,
    }
    DATASET = "MNIST"
    N_SAMPLES = 10000
    K = 3
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

    with Experiment(ExperimentTypes.VAEGeneration).new_run("interpolation") as run:
        mlflow.log_params(hparams)
        mlflow.log_params({"ALPHA": ALPHA, "K": K})
        mlflow.log_metric("N_SAMPLES", N_SAMPLES)
        print("Encoding")
        latents, _, targets = next(iter(encoded_fetcher))
        pca = PCA(2).fit(latents) if latents.size(1) > 2 else None
        visualize_latents(latents, pca, targets, color_by_target=True, img_name="encoded")

        print("Interpolating")
        interpolation = Interpolation(alpha=ALPHA, k=K, return_indices=True)
        inter_latents, inter_targets, indices = interpolation(latents, targets)
        inter_latents = inter_latents.reshape(-1, inter_latents.size(-1))
        inter_targets = inter_targets.reshape(-1, inter_targets.size(-1))
        indices = indices.reshape(-1)
        visualize_latents(inter_latents, pca, inter_targets, color_by_target=True, img_name="interpolated")

        fakes_set = vae.decode_dataset(TensorDataset(inter_latents, inter_targets))
        fakes_loader = DataLoader(fakes_set, batch_size=20 * K)

        print("Real - Fake")
        visualize_real_fake_images(reals, next(iter(fakes_loader))[0], n=20, k=K, indices=None)

        print("Interpolation across dimension")
        inter_dataset = TensorDataset(*interpolate_along_dimension(latents, targets, 20))
        decoded_loader = DataLoader(vae.decode_dataset(inter_dataset), batch_size=20)
        for i, (x_, _) in enumerate(decoded_loader):
            visualize_images(x_, n=x_.size(0), rows=x_.size(0), img_name=f"interpolated-{i}")

