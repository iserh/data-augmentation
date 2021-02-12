from typing import Any, Dict, Tuple
from torch import Tensor
from vae.generation import augmentations
from vae.generation.interpolation import Interpolation
from vae.generation.noise import Noise
import torch


def augment_data(
    original_latents: Tensor,
    original_log_variances: Tensor,
    original_targets: Tensor,
    augmentation: str,
    **kwargs: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Tensor]:
    if augmentation == augmentations.INTERPOLATION:
        # augment using interpolation
        interpolation = Interpolation(alpha=kwargs["alpha"], k=kwargs["k"], return_indices=True)
        latents, targets, indices = interpolation(original_latents, original_targets)

    elif augmentation == augmentations.EXTRAPOLATION:
        # augment using extrapolation
        interpolation = Interpolation(alpha=-kwargs["alpha"], k=kwargs["k"], return_indices=True)
        latents, targets, indices = interpolation(original_latents, original_targets)

    elif augmentation == augmentations.NOISE:
        # augment using noise
        noise = Noise(alpha=kwargs["alpha"], k=kwargs["k"], std=original_latents.std(), return_indices=True)
        latents, targets, indices = noise(original_latents, original_targets)

    elif augmentation == augmentations.EXTRAPOLATION_NOISE:
        # augment using interpolation and noise
        interpolation = Interpolation(alpha=kwargs["alpha"], k=kwargs["k"], return_indices=True)
        latents, targets, indices = interpolation(original_latents, original_targets)
        noise = Noise(
            alpha=kwargs["alpha2"], k=kwargs["k2"], std=latents.std(), return_indices=True, indices_before=indices
        )
        latents, targets, indices = noise(latents, targets)

    elif augmentation == augmentations.INTERPOLATION_NOISE:
        # augment using interpolation and noise
        interpolation = Interpolation(alpha=-kwargs["alpha"], k=kwargs["k"], return_indices=True)
        latents, targets, indices = interpolation(original_latents, original_targets)
        noise = Noise(
            alpha=kwargs["alpha2"], k=kwargs["k2"], std=latents.std(), return_indices=True, indices_before=indices
        )
        latents, targets, indices = noise(latents, targets)

    elif augmentation == augmentations.FORWARD:
        latents = original_latents.unsqueeze(1).expand(original_latents.size(0), kwargs["k"], *original_latents.size()[1:]).reshape(-1, *original_latents.size()[1:])
        log_vars = original_log_variances.unsqueeze(1).expand(original_log_variances.size(0), kwargs["k"], *original_log_variances.size()[1:]).reshape(-1, *original_log_variances.size()[1:])
        targets = original_targets.unsqueeze(1).expand(original_targets.size(0), kwargs["k"], *original_targets.size()[1:]).reshape(-1, *original_targets.size()[1:])
        indices = torch.arange(0, len(original_latents)).unsqueeze(1).expand(len(original_latents), kwargs["k"]).reshape(-1)
        eps = torch.empty_like(log_vars).normal_()
        latents = eps * (0.5 * log_vars).exp() + latents

    else:
        raise NotImplementedError("Augmentation method not implemented!")

    return latents, targets, indices


if __name__ == "__main__":
    latents = torch.normal(0, 1, size=(5, 2))
    log_vars = torch.rand(size=(5, 2))
    targets = torch.randint(0, 10, size=(5,))

    print(latents.size())
    print(log_vars.size())
    print(targets.size())
    latents, targets, indices = augment_data(latents, log_vars, targets, augmentations.FORWARD, k=3)
    print(latents.size())
    print(targets.size())
    print(indices.size())
    print(indices)
    print(latents)
    print(targets)
