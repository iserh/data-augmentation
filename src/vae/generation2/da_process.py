"""Scripts for augmenting datasets."""
from dataclasses import fields
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset

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
            mlflow.log_param("vae_epochs", self.vae_epochs)
            mlflow.log_param("multi_vae", self.multi_vae)
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

            generated_dataset, origins, others = gen.generate(augmentation, n_generate, **kwargs)
            generated_datasets.append(generated_dataset)

        return generated_datasets
