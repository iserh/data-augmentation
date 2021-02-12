"""Data generation by running forward pass through net."""
from typing import Optional, Tuple

import torch
from torch import Tensor

from vae.visualization import visualize_real_fake_images


class Noise:
    def __init__(
        self, alpha: float, k: int, std: float, return_indices: bool = False, indices_before: Optional[Tensor] = None
    ) -> None:
        self.alpha = alpha
        self.k = k
        self.std = std
        self.return_indices = return_indices
        self.indices_before = indices_before

    def __call__(self, z: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        z_ = z.unsqueeze(1).expand(z.size(0), self.k, *z.size()[1:])
        y_ = y.unsqueeze(1).expand(y.size(0), self.k, *y.size()[1:])
        indices_before = (
            torch.tensor(self.indices_before) if self.indices_before is not None else torch.arange(0, z.size(0), 1)
        )
        new_indices = indices_before.unsqueeze(1).expand(indices_before.size(0), self.k)
        normal = torch.empty_like(z_).normal_(0, self.std)
        z_ = z_ + self.alpha * normal

        if self.return_indices:
            return z_.reshape(-1, *z.size()[1:]), y_.reshape(-1, *y.size()[1:]), new_indices.reshape(-1)
        else:
            return z_.reshape(-1, *z.size()[1:]), y_.reshape(-1, *y.size()[1:])


if __name__ == "__main__":
    import mlflow
    from core.data import MNIST_Dataset
    from sklearn.decomposition import PCA
    from torch.utils.data import DataLoader, TensorDataset

    from utils.data.dataset import LoaderDataset
    from utils.integrations import BackendStore, ExperimentName
    from vae.applied_vae import VAEForDataAugmentation
    from vae.models import MNISTVAE, VAEConfig
    from vae.visualization import visualize_latents, visualize_real_fake_images

    vae_config = VAEConfig(total_epochs=5, epochs=5, z_dim=2, beta=1.0)
    DATASET = "MNIST"
    N_SAMPLES = 1000
    K = 2
    ALPHA = 0.3

    mlflow.set_tracking_uri(BackendStore.MNIST.value)
    mlflow.set_experiment(ExperimentName.VAEGeneration.value)

    dataset = MNIST_Dataset()
    model = MNISTVAE.from_pretrained(vae_config)
    vae = VAEForDataAugmentation(model)

    fetch_loader = DataLoader(dataset, batch_size=512, num_workers=4, shuffle=False)
    fetched_set = LoaderDataset(fetch_loader)
    dataloader = DataLoader(fetched_set, batch_size=N_SAMPLES)
    reals, targets = next(iter(dataloader))
    loaded_dataset = TensorDataset(reals, targets)

    encoded_fetch = vae.encode_dataset(loaded_dataset)
    encoded_fetcher = DataLoader(encoded_fetch, batch_size=N_SAMPLES)

    with mlflow.start_run(run_name="noise") as run:
        mlflow.log_params(vae_config.__dict__)
        mlflow.log_params({"ALPHA": ALPHA, "K": K, "N_SAMPLES": N_SAMPLES})
        print("Encoding")
        latents, targets = next(iter(encoded_fetcher))
        pca = PCA(2).fit(latents) if latents.size(1) > 2 else None
        visualize_latents(latents, pca, targets, color_by_target=True, img_name="encoded")

        print("Noise")
        noise = Noise(alpha=ALPHA, k=K, std=torch.std(latents))
        mod_latents, mod_targets = noise(latents, targets)
        visualize_latents(mod_latents, pca, mod_targets, color_by_target=True, img_name="interpolated")

        fakes_set = vae.decode_dataset(TensorDataset(mod_latents, mod_targets))
        fakes_loader = DataLoader(fakes_set, batch_size=20 * K)

        print("Real - Fake")
        visualize_real_fake_images(reals, next(iter(fakes_loader))[0], n=20, k=K, indices=None)
