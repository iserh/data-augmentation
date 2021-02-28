"""Scripts for augmenting datasets."""
from dataclasses import fields
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from vae.visualization import visualize_latents, visualize_images
from sklearn.decomposition import PCA

from utils.mlflow import mlflow_active, mlflow_available
from vae.models import VAEConfig, VAEForDataAugmentation

from .generator_v2 import GeneratorV2

if mlflow_available():
    import mlflow


class DataAugmentation:
    def __init__(
        self,
        vae_config: VAEConfig,
        vae_epochs: int,
        multi_vae: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.vae_config = vae_config
        self.vae_epochs = vae_epochs
        self.multi_vae = multi_vae
        self.seed = seed

    def augment_datasets(
        self,
        datasets: List[Dataset],
        dataset_info: Dict[str, Any],
        augmentation: str,
        K: int,
        balancing: bool = False,
        **kwargs,
    ) -> List[Dataset]:
        if mlflow_active():
            # log vae config except the model attributes
            mlflow.log_params(
                {"vae_" + f.name: getattr(self.vae_config, f.name) for f in fields(self.vae_config) if f.name != "attr"}
            )
            mlflow.log_params({
                "vae_epochs": self.vae_epochs,
                "multi_vae": self.multi_vae,
                "K": K,
                "balancing": balancing,
            })
            # log augmentation parameters
            mlflow.log_params(kwargs)

        # seeding
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # load generative classifier
        # load vae now if not using multi vae
        if not self.multi_vae:
            vae = VAEForDataAugmentation.from_pretrained(self.vae_config, epochs=self.vae_epochs)

        n = dataset_info["n_classes"]
        total_count = sum(dataset_info["class_counts"])
        # x is the proportion of class i in the dataset
        x = torch.Tensor([count / total_count for count in dataset_info["class_counts"]])
        # L is the amount of data for each class that has to be generated
        L = torch.round((1 - x) / (n - 1) * K).long() if balancing else torch.round(x * K).long()

        print(f"n = {n}")
        print(f"x = {x}")
        print(f"L = {L}")

        generated_datasets = []
        all_latents, all_generated_latents, all_labels, all_generated_labels = [], [], [], []
        for label, n_generate, dataset in zip(dataset_info["classes"], L, datasets):
            if self.multi_vae:
                vae_config_label_i = VAEConfig(**self.vae_config.__dict__)
                vae_config_label_i.attr["label"] = label
                vae = VAEForDataAugmentation.from_pretrained(vae_config_label_i, self.vae_epochs)

            gen = GeneratorV2(
                generative_model=vae,
                dataset=dataset,
                seed=self.seed,
            )

            generated_dataset, latents, labels, generated_latents, origins, others = gen.generate(augmentation, n_generate, **kwargs)
            all_latents.append(latents)
            all_generated_latents.append(generated_latents)
            all_labels.append(labels)
            all_generated_labels.append(generated_dataset.tensors[1])
            self.visualize_class(label, generated_dataset.tensors[0], latents, generated_latents, origins, others)
            generated_datasets.append(generated_dataset)
        
        self.visualize_all(torch.cat(all_latents, dim=0), torch.cat(all_generated_latents, dim=0), torch.cat(all_labels, dim=0),  torch.cat(all_generated_labels, dim=0))
        return generated_datasets
    
    @staticmethod
    def visualize_all(latents: Tensor, generated_latents: Tensor, labels: Tensor, generated_labels: Tensor):
        pca = PCA(2).fit(latents) if latents.size(0) > 2 else None
        visualize_latents(latents, pca, labels, color_by_label=True, filename="original_latents.png")
        visualize_latents(generated_latents, pca, generated_labels, color_by_label=True, filename="generated_latents.png")

    @staticmethod
    def visualize_class(label: int, generated_inputs: Tensor, latents: Tensor, generated_latents: Tensor, origins: Tensor, others: Tensor):
        pca = PCA(2).fit(latents) if latents.size(0) > 2 else None
        visualize_latents(latents, pca, filename=f"original_latents_class_{label}.png")
        visualize_latents(generated_latents, pca, filename=f"generated_latents_class_{label}.png")
        visualize_images(generated_inputs, n=50, heritages=origins, partners=others, cols=5, filename=f"generated_images_class_{label}.png")
