"""Data generation by running forward pass through net."""
from typing import Tuple

import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


class Interpolation:
    def __init__(self, alpha: float, k: int = 3) -> None:
        self.alpha = alpha
        self.k = k
    
    def __call__(self, z: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        # build nearest neighbour tree
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm="ball_tree").fit(z)
        # get indices of k nearest neighbours for each latent vector
        _, indices = nbrs.kneighbors(z)
        # generate k new latents for each original latent vector
        # by interpolating between the k'th nearest neighbour
        z_ = torch.empty((z.size(0) * 3, *z.size()[1:]), device=z.device)
        y_ = torch.empty((y.size(0) * 3, *y.size()[1:]), device=y.device)
        for i in range(z.size(0)):
            # each latent vector generates 'n_neighbor' new latent vectors
            for j, k in enumerate(indices[i]):
                # interpolate between latent vector and the k'th nearest neighbour
                z_[i + j] = (z[k] - z[i]) * self.alpha + z[i]
                # save the target too
                y_[i + j] = y[i]
        # return new modified latents and the corresponding targets as tensors
        return z_, y_


if __name__ == "__main__":
    from utils.mlflow_utils import Experiment, ExperimentTypes, get_run, load_pytorch_model, Roots
    from vae.commons.applied_vae import AppliedVAE
    from vae.commons.visualization import visualize_real_fake_images
    from vae.mnist.dataset import MNISTWrapper
    import mlflow
    from vae.commons.visualization import visualize_latents, visualize_real_fake_images
    from torch.utils.data import DataLoader
    from utils.data import BatchCollector, LambdaDataset, LoaderDataset
    from sklearn.decomposition import PCA

    hparams = {
        "EPOCHS": 100,
        "Z_DIM": 2,
        "BETA": 1.0,
    }
    DATASET = "MNIST"
    N_SAMPLES = 2048

    mlflow.set_tracking_uri(Roots.MNIST.value)

    dataset = MNISTWrapper()
    model = load_pytorch_model(get_run(ExperimentTypes.VAETraining, **hparams), chkpt=hparams["EPOCHS"])
    vae = AppliedVAE(model, cuda=True)

    encoded_loader = DataLoader(vae.encode_dataset(dataset, shuffle=False), batch_size=512, collate_fn=BatchCollector.collate_fn)
    encoded_dataset = LoaderDataset(encoded_loader)

    with Experiment(ExperimentTypes.VAEGeneration).new_run("interpolation") as run:
        print("Encoding")
        latent_loader = DataLoader(encoded_dataset, batch_size=N_SAMPLES, collate_fn=BatchCollector.collate_fn)
        latents, targets = next(iter(latent_loader))
        print(latents.size())
        pca = PCA(2).fit(latents) if latents.size(1) > 2 else None
        visualize_latents(latents, pca, targets, color_by_target=True, img_name="encoded")

        print("Interpolating")
        interpolation = Interpolation(alpha=0.5, k=3)
        interpolated_dataset = LambdaDataset(encoded_dataset, BatchCollector(interpolation))
        interpolated_loader = DataLoader(interpolated_dataset, batch_size=N_SAMPLES * 3, collate_fn=BatchCollector.collate_fn)
        latents, targets = next(iter(interpolated_loader))
        print(latents.size())
        pca = PCA(2).fit(latents) if latents.size(1) > 2 else None
        visualize_latents(latents, pca, targets, color_by_target=True, img_name="interpolated")
