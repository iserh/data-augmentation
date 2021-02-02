"""Data generation by running forward pass through net."""
from typing import Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


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
        z_ = torch.empty((z.size(0), self.k, *z.size()[1:]), device=z.device, dtype=z.dtype)
        y_ = torch.empty((y.size(0), self.k, *y.size()[1:]), device=y.device, dtype=y.dtype)
        for i in range(z.size(0)):
            # each latent vector generates 'n_neighbor' new latent vectors
            for j, k in enumerate(indices[i]):
                # interpolate between latent vector and the k'th nearest neighbour
                z_[i, j] = (z[k] - z[i]) * self.alpha + z[i]
                # save the target too
                y_[i, j] = y[i]
        z_ = z_.reshape(-1, z.size(-1))
        y_ = y_.flatten()
        # return new modified latents and the corresponding targets as tensors
        return (z_, y_, indices.reshape(-1)) if self.return_indices else (z_, y_)


def interpolate_along_class(latents: Tensor, targets: Tensor, n_steps: int) -> Tensor:
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


def interpolate_along_dimension(z: Tensor, n_steps: int) -> Tensor:
    interpolated = []
    for dim in range(z.size(0)):
        other_dims = z.unsqueeze(0).expand((n_steps, z.size(0))).clone()
        x = np.linspace(z[dim] - 3, z[dim] + 3, n_steps)
        other_dims[:, dim] = torch.Tensor(x)
        interpolated.append(other_dims)
    return torch.cat(interpolated, dim=0)


if __name__ == "__main__":
    import mlflow
    from core.data import MNIST_Dataset
    from torch.utils.data import DataLoader, TensorDataset

    from utils.data import LoaderDataset
    from utils.integrations import BackendStore, ExperimentName
    from vae.applied_vae import VAEForDataAugmentation
    from vae.models import MNISTVAE, VAEConfig
    from vae.visualization import visualize_images, visualize_latents, visualize_real_fake_images

    DATASET = "MNIST"
    N_SAMPLES = 256
    K = 3
    ALPHA = 0.5

    mlflow.set_tracking_uri(BackendStore[DATASET].value)
    mlflow.set_experiment(ExperimentName.VAEGeneration.value)

    dataset = MNIST_Dataset()
    vae_config = VAEConfig(epochs=5, checkpoint=5, z_dim=2, beta=1.0)
    model = MNISTVAE.from_pretrained(vae_config)
    vae = VAEForDataAugmentation(model)

    fetch_loader = DataLoader(dataset, batch_size=512, num_workers=4, shuffle=False)
    fetched_set = LoaderDataset(fetch_loader)
    dataloader = DataLoader(fetched_set, batch_size=N_SAMPLES)
    reals, targets = next(iter(dataloader))
    loaded_dataset = TensorDataset(reals, targets)

    encoded_fetch = vae.encode_dataset(loaded_dataset)
    encoded_fetcher = DataLoader(encoded_fetch, batch_size=N_SAMPLES)

    with mlflow.start_run(run_name="interpolation"):
        mlflow.log_params(vae_config.__dict__)
        mlflow.log_params({"ALPHA": ALPHA, "K": K})
        mlflow.log_metric("N_SAMPLES", N_SAMPLES)
        print("Encoding")
        latents, targets = next(iter(encoded_fetcher))
        pca = PCA(2).fit(latents) if latents.size(1) > 2 else None
        visualize_latents(latents, pca, targets, color_by_target=True, img_name="encoded")

        print("Interpolating")
        interpolation = Interpolation(alpha=ALPHA, k=K, return_indices=True)
        inter_latents, inter_targets, indices = interpolation(latents, targets)
        visualize_latents(inter_latents, pca, inter_targets, color_by_target=True, img_name="interpolated")

        fakes_set = vae.decode_dataset(TensorDataset(inter_latents, inter_targets))
        fakes_loader = DataLoader(fakes_set, batch_size=20 * K)

        print("Real - Fake")
        visualize_real_fake_images(reals, next(iter(fakes_loader))[0], n=20, k=K, indices=indices)

        print("Interpolation across class")
        inter_dataset = TensorDataset(*interpolate_along_class(latents, targets, n_steps=20))
        decoded_loader = DataLoader(vae.decode_dataset(inter_dataset), batch_size=20)
        for i, (x_, _) in enumerate(decoded_loader):
            visualize_images(x_, n=x_.size(0), rows=x_.size(0), img_name=f"interpolated-class-{i}")

        print("Interpolation across dimension")
        inter_dataset = TensorDataset(interpolate_along_dimension(latents[0], n_steps=20))
        decoded_loader = DataLoader(vae.decode_dataset(inter_dataset), batch_size=20)
        for i, (x_,) in enumerate(decoded_loader):
            visualize_images(x_, n=x_.size(0), rows=x_.size(0), img_name=f"interpolated-dim-{i}")
