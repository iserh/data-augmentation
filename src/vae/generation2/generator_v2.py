"""Scripts for augmenting datasets."""
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from vae.generation import augmentations
from vae.generation.distribution import apply_distribution
from vae.generation.interpolation import apply_extrapolation, apply_interpolation
from vae.generation.noise import add_noise, normal_noise
from vae.generation.reparametrization import apply_reparametrization
from vae.models import VAEForDataAugmentation

from generative_classifier.models import GenerativeClassifierModel
from vae.visualization.vis import visualize_images

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
    ) -> None:
        self.generative_model = generative_model
        self.generative_classifier = generative_classifier
        self.dataset = dataset
        self.seed = seed

    def generate(self, augmentation: str, n: int, **kwargs) -> Tuple[TensorDataset, Tensor, Tensor, Tensor, Tensor]:
        # seeding
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # get the data from the dataset
        inputs, labels = next(iter(DataLoader(self.dataset, batch_size=len(self.dataset))))
        # encode to latent vectors
        latents, log_vars, _ = self.generative_model.encode_dataset(self.dataset).tensors

        all_good = False
        final_inputs, final_labels, final_generated_latents, final_origins, final_others = [], [], [], [], []
        i = 0
        while not all_good:
            i += 1
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

            decoded_inputs = self.generative_model.decode_dataset(TensorDataset(generated_latents, sample_labels)).tensors[0]

            if self.generative_classifier is not None:
                decoded_loader = DataLoader(TensorDataset(decoded_inputs), batch_size=512)
                output = [self.generative_classifier(x).prediction.view(-1) for (x,) in decoded_loader]
                is_real = torch.cat(output, dim=0).bool()
                if is_real.all():
                    all_good = True
                else:
                    visualize_images(decoded_inputs[~is_real], n=50, filename=f"noise-iteration-{i}.png")
                final_inputs.append(decoded_inputs[is_real])
                final_labels.append(sample_labels[is_real])
                final_generated_latents.append(generated_latents[is_real])
                final_origins.append(origins[is_real] if origins is not None else None)
                final_others.append(others[is_real] if others is not None else None)
                n = n - is_real.sum()
            else:
                final_inputs, final_labels = decoded_inputs, sample_labels
                all_good = True
        
        final_inputs = torch.cat(final_inputs, dim=0)
        final_labels = torch.cat(final_labels, dim=0)
        final_generated_latents = torch.cat(final_generated_latents, dim=0)
        final_origins = torch.cat(final_origins, dim=0) if final_origins[0] is not None else None
        final_others = torch.cat(final_others, dim=0) if final_others[0] is not None else None

        return TensorDataset(final_inputs, final_labels), inputs, labels, latents, final_generated_latents, final_origins, final_others
