"""Scripts for augmenting datasets."""
from dataclasses import fields
from typing import Any, Dict, Optional, Tuple, Union

import mlflow
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

from vae.generation import augmentations
from vae.generation.interpolation import Interpolation
from vae.generation.noise import Noise
from vae.models import VAEConfig, VAEForDataAugmentation
from vae.visualization import visualize_images, visualize_latents, visualize_real_fake_images


def augment_dataset_using_single_vae(
    dataset: Dataset,
    vae_config: VAEConfig,
    vae_epochs: int,
    augmentation_method: str,
    augmentation_params: Dict[str, Union[float, int]],
    seed: Optional[int] = None,
) -> Dataset:
    """Augment a dataset using a single VAE pretrained for this dataset.

    Args:
        dataset (Dataset): The dataset to augment
        vae_config (VAEConfig): The configuration of the VAE to use
        vae_epochs (int): Number of epochs the VAE was trained
        augmentation_method (str): The augmentation to apply
        augmentation_params (Dict[str, Union[float, int]]): Parameters for the augmentation method
        seed (Optional[int]): Seed for reproducibility

    Raises:
        NotImplementedError: If the augmentation method is invalid or not implemented

    Returns:
        Dataset: The augmented dataset
    """
    if augmentation_method == augmentations.RANDOM_NOISE:
        raise NotImplementedError("Single VAE augmentation cannot be used with RANDOM_NOISE augmentation method!")
    # seeding
    if seed is not None:
        torch.manual_seed(seed)

    # number of generated examples to plot
    n_img_plots = 50

    # log vae config except the model attributes
    mlflow.log_params({"vae_" + f.name: getattr(vae_config, f.name) for f in fields(vae_config) if f.name != "attr"})
    # log augmentation parameters
    mlflow.log_params(augmentation_params)

    # extract the real images and labels from the dataset
    reals = next(iter(DataLoader(dataset, batch_size=len(dataset))))[0]

    # *** Data augmentation ***

    # load vae model
    vae = VAEForDataAugmentation.from_pretrained(vae_config, epochs=vae_epochs)
    # encode dataset
    original_latents, original_log_vars, original_targets = vae.encode_dataset(dataset).tensors
    # augment data - get augmented latents, targets and the indices used for augmentation (for visualization)
    aug_latents, aug_targets, aug_indices = augment_latents(
        original_latents, original_log_vars, original_targets, augmentation_method, **augmentation_params
    )
    # decode augmented latents
    decoded = vae.decode_dataset(TensorDataset(aug_latents, aug_targets))

    # ----------------------------------------

    # pca for 2d view
    pca = PCA(2).fit(original_latents)
    # visualize encoded latents
    visualize_latents(
        original_latents, pca, targets=original_targets, color_by_target=True, img_name="original_latents"
    )
    # visualize augmented latents
    visualize_latents(aug_latents, pca, targets=aug_targets, color_by_target=True, img_name="augmented_latents")
    # visualize the fake images, compared to their originals used for generating them
    visualize_real_fake_images(
        reals,
        decoded.tensors[0],
        n=n_img_plots,
        k=augmentation_params["k"] * augmentation_params.get("k2", 1),
        indices=aug_indices,
        cols=5,
    )

    # concat the original and the augmented dataset
    concat_dataset = ConcatDataset([dataset, decoded])
    print(f"Augmented dataset from {len(dataset)} samples to {len(concat_dataset)}")
    # return the concatenated dataset
    return concat_dataset


def augment_dataset_using_per_class_vaes(
    dataset: Dataset,
    vae_config: VAEConfig,
    vae_epochs: int,
    augmentation_method: str,
    augmentation_params: Dict[str, Union[float, int]],
    seed: Optional[int] = None,
) -> Dataset:
    """Augment a dataset using one VAE for each class label.

    Args:
        dataset (Dataset): The dataset to augment
        vae_config (VAEConfig): The configuration of the VAE to use
        vae_epochs (int): Number of epochs the VAE was trained
        augmentation_method (str): The augmentation to apply
        augmentation_params (Dict[str, Union[float, int]]): Parameters for the augmentation method
        seed (Optional[int]): Seed for reproducibility

    Returns:
        Dataset: The augmented dataset
    """
    # seeding
    if seed is not None:
        torch.manual_seed(seed)

    # number of generated examples to plot
    n_img_plots = 50

    # log vae config except the model attributes
    mlflow.log_params({"vae_" + f.name: getattr(vae_config, f.name) for f in fields(vae_config) if f.name != "attr"})
    # log augmentation parameters
    mlflow.log_params(augmentation_params)

    # extract the real images and labels from the dataset
    reals, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))

    # *** Data augmentation ***

    # get the unique labels
    unique_labels = torch.unique(labels, sorted=True).tolist()
    # augmented data of each label will be appended to this list
    datasets_per_label = []
    # save some information about the augmented datasets
    dataset_info = pd.DataFrame(columns=["LABEL", "ORIGINAL", "AUGMENTED"])

    # iterate over each unique label, augmenting the data of each label with seperate VAEs
    for label in unique_labels:
        # get mask for current label
        mask = labels == label
        # set vae config to current label
        vae_config.attr = {"label": label}
        # load vae model
        vae = VAEForDataAugmentation.from_pretrained(vae_config, epochs=vae_epochs)
        # create tensor dataset for vae
        label_dataset = TensorDataset(reals[mask], labels[mask])
        # encode dataset
        original_latents, original_log_vars, original_targets = vae.encode_dataset(label_dataset).tensors
        # augment data - get augmented latents, targets and the indices used for augmentation (for visualization)
        aug_latents, aug_targets, aug_indices = augment_latents(
            original_latents, original_log_vars, original_targets, augmentation_method, **augmentation_params
        )
        # decode augmented latents
        decoded = vae.decode_dataset(TensorDataset(aug_latents, aug_targets))

        # ----------------------------------------

        # pca for 2d view
        pca = PCA(2).fit(original_latents) if original_latents.size(-1) > 2 else None
        # visualize encoded latents
        visualize_latents(
            original_latents,
            pca,
            targets=original_targets,
            color_by_target=True,
            img_name=f"original_latents_label_{label}",
        )
        # visualize augmented latents
        visualize_latents(
            aug_latents, pca, targets=aug_targets, color_by_target=True, img_name=f"augmented_latents_label_{label}"
        )
        # visualize the fake images, compared to their originals used for generating them
        # if augmentation method is RANDOM_NOISE, no comparison is possible so just visualize the fake images
        if augmentation_method != augmentations.RANDOM_NOISE:
            visualize_real_fake_images(
                label_dataset.tensors[0],
                decoded.tensors[0],
                n=n_img_plots,
                k=augmentation_params["k"] * augmentation_params.get("k2", 1),
                indices=aug_indices,
                cols=15,
                img_name=f"real_fake_label_{label}",
            )
        else:
            visualize_images(decoded.tensors[0], n=n_img_plots, img_name=f"fakes_label_{label}", cols=5)

        # append augmented data of this label to the dataset list
        datasets_per_label.append(ConcatDataset([label_dataset, decoded]))
        dataset_info.loc[label] = [label, len(label_dataset), len(datasets_per_label[-1])]

    # log the statistics about the datasets (number of samples, ...)
    print(dataset_info)
    # concat the original and the augmented dataset
    return ConcatDataset(datasets_per_label)


def augment_latents(
    original_latents: Tensor,
    original_log_variances: Tensor,
    original_labels: Tensor,
    augmentation_method: str,
    **kwargs: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Augment given latent vectors by the specified augmentation method.

    Generates one or more latent vector for each given latent vector.

    Args:
        original_latents (Tensor): Original latent vectors of the encodings
        original_log_variances (Tensor): Original calculated logarithmics of variances of the encodings
        original_labels (Tensor): Original labels associated with the original latent vectors & variances
        augmentation_method (str): The augmentation method to apply
        **kwargs (Dict[str, Any]): Parameters for the augmentation method

    Raises:
        NotImplementedError: If the specified augmentation method is not implemented or unknown.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The new generated latent vectors,
        the associated labels and the indices of the original latents, that were used for generation.
    """
    # *** Interpolation ***
    if augmentation_method == augmentations.INTERPOLATION:
        # initialize interpolation function
        interpolation = Interpolation(alpha=kwargs["alpha"], k=kwargs["k"], return_indices=True)
        # interpolate between k nearest neighbours
        latents, labels, indices = interpolation(original_latents, original_labels)

    # *** Extrapolation ***
    elif augmentation_method == augmentations.EXTRAPOLATION:
        # initialize interpolation function with negative alpha -> extrapolation
        interpolation = Interpolation(alpha=-kwargs["alpha"], k=kwargs["k"], return_indices=True)
        # interpolate between k nearest neighbours
        latents, labels, indices = interpolation(original_latents, original_labels)

    # *** Noise ***
    elif augmentation_method == augmentations.NOISE:
        # initialize noise function
        noise = Noise(alpha=kwargs["alpha"], k=kwargs["k"], std=original_latents.std(), return_indices=True)
        # add noise to latents
        latents, labels, indices = noise(original_latents, original_labels)

    # *** Interpolation & Noise ***
    elif augmentation_method == augmentations.EXTRAPOLATION_NOISE:
        # initialize interpolation function
        interpolation = Interpolation(alpha=kwargs["alpha"], k=kwargs["k"], return_indices=True)
        # interpolate between k nearest neighbours
        latents, labels, indices = interpolation(original_latents, original_labels)
        # initialize noise function, give it the indices of interpolation keep track of the images used for generation
        noise = Noise(
            alpha=kwargs["alpha2"], k=kwargs["k2"], std=latents.std(), return_indices=True, indices_before=indices
        )
        # add noise to latents
        latents, labels, indices = noise(latents, labels)

    # *** Extrapolation & Noise ***
    elif augmentation_method == augmentations.INTERPOLATION_NOISE:
        # initialize interpolation function with negative alpha -> extrapolation
        interpolation = Interpolation(alpha=-kwargs["alpha"], k=kwargs["k"], return_indices=True)
        # interpolate between k nearest neighbours
        latents, labels, indices = interpolation(original_latents, original_labels)
        # initialize noise function, give it the indices of interpolation keep track of the images used for generation
        noise = Noise(
            alpha=kwargs["alpha2"], k=kwargs["k2"], std=latents.std(), return_indices=True, indices_before=indices
        )
        # add noise to latents
        latents, labels, indices = noise(latents, labels)

    # *** Simple forward pass ***
    elif augmentation_method == augmentations.FORWARD:
        # duplicate the original latents to reach the desired number of generated samples
        latents = (
            original_latents.unsqueeze(1)
            .expand(original_latents.size(0), kwargs["k"], *original_latents.size()[1:])
            .reshape(-1, *original_latents.size()[1:])
        )
        # duplicate the original log variances to reach the desired number of generated samples
        log_vars = (
            original_log_variances.unsqueeze(1)
            .expand(original_log_variances.size(0), kwargs["k"], *original_log_variances.size()[1:])
            .reshape(-1, *original_log_variances.size()[1:])
        )
        # duplicate the original labels to reach the desired number of generated samples
        labels = (
            original_labels.unsqueeze(1)
            .expand(original_labels.size(0), kwargs["k"], *original_labels.size()[1:])
            .reshape(-1, *original_labels.size()[1:])
        )
        # indices are just each index of a latent vector duplicated to reach the desired number of examples
        indices = (
            torch.arange(0, len(original_latents)).unsqueeze(1).expand(len(original_latents), kwargs["k"]).reshape(-1)
        )
        # create a normal distributed epsilon
        eps = torch.empty_like(log_vars).normal_()
        # multiply epsilon with variance and add mean to sample from a
        # gaussian with mean, variance corresponding to the encoding
        latents = eps * (0.5 * log_vars).exp() + latents

    # *** Random Noise ***
    # This method is only possible if all labels are the same
    elif augmentation_method == augmentations.RANDOM_NOISE:
        # sample random latents from a normal distribution
        latents = torch.empty((kwargs["k"] * len(original_latents), *original_latents.size()[1:])).normal_(0, 1)
        # duplicate the labels to reach the desired number of examples
        labels = torch.tensor([original_labels[0]] * kwargs["k"] * len(original_labels))
        # in this method latents are randomly created, so no original latents were used for generation -> indices = None
        indices = None

    # *** Not Implemented ***
    else:
        raise NotImplementedError("Augmentation method not implemented!")

    # return the generated latents, labels and indices of the original latents used for generation
    return latents, labels, indices
