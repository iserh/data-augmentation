"""Scripts for augmenting datasets."""
from dataclasses import fields
from typing import Optional, Tuple, Union

import torch
from sklearn.decomposition import PCA
from torch import Tensor, LongTensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

from matplotlib.pyplot import close
from vae.models import VAEConfig, VAEForDataAugmentation
from vae.visualization import visualize_latents, visualize_images
from . import augmentations
from .reparametrization import apply_reparametrization
from .interpolation import apply_interpolation, apply_extrapolation
from .noise import add_noise, normal_noise
from .distribution import apply_distribution

implementations = {
    augmentations.INTERPOLATION: apply_interpolation,
    augmentations.EXTRAPOLATION: apply_extrapolation,
    augmentations.REPARAMETRIZATION: apply_reparametrization,
    augmentations.ADD_NOISE: add_noise,
    augmentations.NORMAL_NOISE: normal_noise,
    augmentations.DISTRIBUTION: apply_distribution,
}


class Generator:
    def __init__(
        self,
        vae_config: VAEConfig,
        vae_epochs: int,
        multi_vae: bool = False,
        seed: Optional[int] = None,
        no_mlflow: bool = False,
    ) -> None:
        self.vae_config = vae_config
        self.vae_epochs = vae_epochs
        self.multi_vae = multi_vae
        self.seed = seed

        # enable mlflow
        self.mlflow_enabled = not no_mlflow
        if self.mlflow_enabled:
            import mlflow

            self.mlflow = mlflow

    def augment_dataset(self, dataset: Dataset, augmentation: str, **kwargs) -> ConcatDataset:
        if self.mlflow_enabled:
            self.mlflow.log_param("original_dataset_size", len(dataset))
            # log vae config except the model attributes
            self.mlflow.log_params(
                {"vae_" + f.name: getattr(self.vae_config, f.name) for f in fields(self.vae_config) if f.name != "attr"}
            )
            self.mlflow.log_param("vae_epochs", self.vae_epochs)
            self.mlflow.log_param("multi_vae", self.multi_vae)
            # log augmentation parameters
            self.mlflow.log_params(kwargs)

        # seeding
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # augment data
        if self.multi_vae:
            augmented = self._multi_vae(dataset, augmentation, **kwargs)
        else:
            augmented = self._single_vae(dataset, augmentation, **kwargs)
        
        _, class_counts = torch.unique(augmented.tensors[1], sorted=True, return_counts=True)
        print("----------------------------------------")
        print(f"class_counts: {class_counts}, ({class_counts.sum()} total)")
        print(f"x = {torch.Tensor([count / augmented.tensors[1].size(0) for count in class_counts])}")

        # concat the original and the augmented dataset
        concat_dataset = ConcatDataset([dataset, augmented])
        print("----------------------------------------")
        print(f"Augmented dataset from {len(dataset)} samples to {len(concat_dataset)}")
        labels = next(iter(DataLoader(concat_dataset, batch_size=len(concat_dataset))))[1]
        _, class_counts = torch.unique(labels, sorted=True, return_counts=True)
        print("----------------------------------------")
        print(f"class_counts: {class_counts}, ({class_counts.sum()} total)")
        print(f"x = {torch.Tensor([count / augmented.tensors[1].size(0) for count in class_counts])}")
        # return the concatenated dataset
        return concat_dataset

    def _single_vae(self, dataset: Dataset, augmentation: str, **kwargs) -> TensorDataset:
        # load vae model
        vae = VAEForDataAugmentation.from_pretrained(self.vae_config, epochs=self.vae_epochs)
        real_images, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))
        # encode dataset
        latents, log_vars, _ = vae.encode_dataset(dataset).tensors
        # augment data - get augmented latents, targets and the indices used for augmentation (for visualization)
        
        new_latents, new_labels, heritages, partners = apply_augmentation(
            real_images,
            labels,
            latents,
            log_vars,
            augmentation,
            return_heritages_partners=True,
            **kwargs,
        )

        # decode augmented latents
        decoded = vae.decode_dataset(TensorDataset(new_latents, new_labels))

        # visualize
        self._visualize(
            real_images,
            latents,
            new_latents,
            labels,
            new_labels,
            decoded.tensors[0],
            heritages,
            partners,
        )

        return decoded

    def _multi_vae(self, dataset: Dataset, augmentation: str, **kwargs) -> TensorDataset:
        real_images, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))

        # *** Encoding ***

        latents, log_vars = [], []
        for cls in torch.unique(labels, sorted=True).tolist():
            mask = labels == cls
            self.vae_config.attr["label"] = cls
            # load vae model
            vae = VAEForDataAugmentation.from_pretrained(self.vae_config, epochs=self.vae_epochs)
            # encode dataset
            z, log_v, _ = vae.encode_dataset(
                TensorDataset(real_images[mask], labels[mask])
            ).tensors
            latents.append(z)
            log_vars.append(log_v)

        latents = torch.cat(latents, dim=0)
        log_vars = torch.cat(log_vars, dim=0)

        # *** Augmentation ***

        # augment data
        new_latents, new_labels, heritages, partners = apply_augmentation(
            real_images,
            labels,
            latents,
            log_vars,
            augmentation,
            return_heritages_partners=True,
            **kwargs,
        )

        # *** Decoding ***

        # decode augmented latents
        decoded = vae.decode_dataset(TensorDataset(new_latents, new_labels))

        # visualize
        self._visualize(
            real_images,
            latents,
            new_latents,
            labels,
            new_labels,
            decoded.tensors[0],
            heritages,
            partners,
        )

        return decoded
    
    def _visualize(
        self,
        real_images: Tensor,
        latents: Tensor,
        new_latents: Tensor,
        labels: Tensor,
        new_labels: Tensor,
        new_decoded: Tensor,
        heritages: Tensor,
        partners: Tensor,
    ) -> None:
        # number of generated examples to plot
        n_img_plots = 50
        # pca for 2d view
        pca = PCA(2).fit(latents) if latents.size(1) > 2 else None
        # visualize encoded latents
        fig = visualize_latents(latents, pca, labels=labels, color_by_label=True)
        if self.mlflow_enabled:
            self.mlflow.log_figure(fig, "original_latents.png")
            close()
        else:
            fig.show()
        # visualize augmented latents
        fig = visualize_latents(new_latents, pca, labels=new_labels, color_by_label=True)
        if self.mlflow_enabled:
            self.mlflow.log_figure(fig, "augmented_latents.png")
            close()
        else:
            fig.show()
        # plot images of each class
        for cls in torch.unique(new_labels, sorted=True).tolist():
            mask = new_labels == cls
            # visualize the real images
            fig = visualize_images(
                images=real_images[labels == cls],
                n=n_img_plots,
                cols=n_img_plots // 10,
                img_title=f"Original class {cls}",
            )
            if self.mlflow_enabled:
                self.mlflow.log_figure(fig, f"Original_Images_class-{cls}.png")
                close()
            else:
                fig.show()
            # visualize the fake images, compared to their originals used for generating them
            fig = visualize_images(
                images=new_decoded[mask],
                n=n_img_plots,
                heritages=heritages[mask] if heritages is not None else None,
                partners=partners[mask] if partners is not None else None,
                cols=n_img_plots // 10,
                img_title=f"Generated class {cls}",
            )
            if self.mlflow_enabled:
                self.mlflow.log_figure(fig, f"Heritages_Partners_class-{cls}.png")
                close()
            else:
                fig.show()


def apply_augmentation(
    real_images: Tensor,
    labels: Tensor,
    latents: Tensor,
    log_vars: Tensor,
    augmentation: str,
    K: int,
    balance: bool = False,
    return_heritages_partners: bool = False,
    **kwargs,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    classes, class_counts = torch.unique(labels, sorted=True, return_counts=True)
    # n is the number of classes
    n = classes.size(0)
    # x is the proportion of class i in the dataset
    x = torch.Tensor([count / labels.size(0) for count in class_counts])
    # L is the amount of data for each class that has to be generated
    L = torch.round((1 - x) / (n - 1) * K).long() if balance else torch.round(x * K).long()
    # std deviation across the whole dataset (used for noise i.e.)
    kwargs["std"] = kwargs.get("std", latents.std())
    print(f"classes: {classes}")
    print(f"class_counts: {class_counts}, ({class_counts.sum()} total)")
    print(f"n = {n}")
    print(f"x = {x}")
    print(f"L = {L}")

    # new latents, labels and the root latents used for generating them
    new_latents, new_labels, partners, heritages = [], [], [], []
    for i, cls in enumerate(classes.tolist()):
        # get indices of the labels that equal cls
        mask = labels == cls
        # select the corresponding latents
        latents_cls = latents[mask]
        # select the corresponding log_vars
        log_vars_cls = log_vars[mask]
        # select the corresponding real images
        real_images_cls = real_images[mask]
        # select L[i] random indices for choosing the latents to augment
        idx = torch.randint(0, latents_cls.size(0), size=(L[i],))
        latents_cls = latents_cls[idx]
        log_vars_cls = log_vars_cls[idx]
        real_images_cls = real_images_cls[idx]
        # augment latents
        augmented_cls, heritages_cls, partners_cls = implementations[augmentation](real_images_cls, latents_cls, log_vars_cls, unique_latents=latents[mask], unique_reals=real_images[mask], **kwargs)
        new_latents.append(augmented_cls)
        partners.append(partners_cls)
        heritages.append(heritages_cls)
        new_labels.append(LongTensor([cls] * L[i]))

    if return_heritages_partners:
        return (
            torch.cat(new_latents, dim=0),
            torch.cat(new_labels, dim=0),
            torch.cat(heritages, dim=0) if heritages[0] is not None else None,
            torch.cat(partners, dim=0) if partners[0] is not None else None,
        )
    else:
        return torch.cat(new_latents, dim=0), torch.cat(new_labels, dim=0)
