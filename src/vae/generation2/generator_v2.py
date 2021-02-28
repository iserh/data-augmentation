"""Scripts for augmenting datasets."""
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from generative_classifier.models import GenerativeClassifierModel
from vae.models import VAEForDataAugmentation
from vae.generation import augmentations
from vae.generation.reparametrization import apply_reparametrization
from vae.generation.interpolation import apply_interpolation, apply_extrapolation
from vae.generation.noise import add_noise, normal_noise
from vae.generation.distribution import apply_distribution

implementations = {
    augmentations.INTERPOLATION: apply_interpolation,
    augmentations.EXTRAPOLATION: apply_extrapolation,
    augmentations.REPARAMETRIZATION: apply_reparametrization,
    augmentations.ADD_NOISE: add_noise,
    augmentations.NORMAL_NOISE: normal_noise,
    augmentations.DISTRIBUTION: apply_distribution,
}


class GeneratorV2:
    def __init__(
        self,
        generative_model: VAEForDataAugmentation,
        dataset: Dataset,
        generative_classifier: Optional[GenerativeClassifierModel] = None,
        seed: Optional[int] = None,
        no_mlflow: bool = False,
    ) -> None:
        self.generative_model = generative_model
        self.generative_classifier = generative_classifier
        self.dataset = dataset
        self.seed = seed

    def generate(self, augmentation: str, n: int, **kwargs) -> Tuple[TensorDataset, Tensor, Tensor]:
        # seeding
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # get the data from the dataset
        inputs, labels = next(iter(DataLoader(self.dataset, batch_size=len(self.dataset))))
        # encode to latent vectors
        latents, log_vars, _ = self.generative_model.encode_dataset(self.dataset).tensors

        # choose n examples for augmentation
        sample_idx = torch.randint(0, len(self.dataset), size=(n,))
        sample_inputs = inputs[sample_idx]
        sample_latents = latents[sample_idx]
        sample_log_vars = log_vars[sample_idx]
        sample_labels = labels[sample_idx]

        # augment latent vectors, collect origins and others (if they exist in the augmentation method)
        generated_latents, origins, others = implementations[augmentation](
            sample_inputs, sample_latents, sample_log_vars, unique_latents=latents, unique_reals=inputs, **kwargs
        )

        return TensorDataset(generated_latents, sample_labels), origins, others
