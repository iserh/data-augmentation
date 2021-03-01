from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from yaml.loader import SafeLoader

from utils.data import get_dataset

data_path = "datasets/splitted"


def split_datasets(dataset_name: str, reduce: Optional[int] = None, others: bool = True, balancing: bool = True, seed: Optional[int] = None) -> None:
    # seed
    if seed is not None:
        torch.manual_seed(seed)
    # load dataset
    dataset = get_dataset(dataset_name, train=True)
    # optionally reduce size of the dataset
    if reduce is not None and not balancing:
        dataset = Subset(dataset, torch.randperm(len(dataset))[:reduce])
    # extract data from the dataset
    inputs, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))

    if reduce is not None and balancing:
        unique_labels = torch.unique(labels, sorted=True)
        inputs = torch.cat([inputs[labels == label][:reduce // len(unique_labels)] for label in unique_labels])
        labels = torch.cat([labels[labels == label][:reduce // len(unique_labels)] for label in unique_labels])

    # get classes and class counts
    classes, class_counts = torch.unique(labels, sorted=True, return_counts=True)

    # create output dir
    path = Path(data_path) / dataset_name
    (path / "train").mkdir(exist_ok=True, parents=True)

    # save info about the dataset
    with open(path / "info.yml", "w") as yml_file:
        yaml.dump(
            {
                "n_classes": len(classes),
                "classes": classes.tolist(),
                "class_counts": class_counts.tolist(),
            },
            yml_file,
        )

    # iterate over classes
    for label in classes.tolist():
        mask_class = labels == label
        mask_other = labels != label
        labels_class = labels[mask_class]
        labels_other = labels[mask_other]
        inputs_class = inputs[mask_class]
        inputs_other = inputs[mask_other]

        if others:
            # select amount of others to reach a 20% part in total
            n_others = len(labels_class) // 4
            others_idx = torch.randperm(len(labels_other))[:n_others]
            # combine all class inputs/labels with the 20% part of other classes
            inputs_class = torch.cat([inputs_class, inputs_other[others_idx]])
            labels_class = torch.cat([labels_class, labels_other[others_idx]])

        # save this dataset
        torch.save(TensorDataset(inputs_class, labels_class), path / "train" / f"class-{label}.pt")


def load_splitted_datasets(dataset_name: str, others: bool = True) -> Tuple[List[TensorDataset], Dict[str, Any]]:
    path = Path(data_path) / dataset_name
    if not path.exists():
        raise FileNotFoundError("Dataset does not exist.")

    with open(path / "info.yml", "r") as yml_file:
        info = yaml.load(yml_file, Loader=SafeLoader)

    datasets = [torch.load(path / "train" / f"class-{label}.pt") for label in info["classes"]]

    if not others:
        datasets = [TensorDataset(*ds[ds.tensors[1] == label]) for label, ds in zip(info["classes"], datasets)]

    return datasets, info


def load_unsplitted_dataset(dataset_name: str) -> Tuple[ConcatDataset, Dict[str, Any]]:
    datasets, info = load_splitted_datasets(dataset_name, others=False)
    return ConcatDataset(datasets), info


if __name__ == "__main__":
    DATASET = "MNIST"
    split_datasets(DATASET, reduce=50, others=True, seed=1337)
    datasets, _ = load_splitted_datasets(DATASET, others=False)
    print(", ".join([str(len(ds)) for ds in datasets]))
    datasets, _ = load_splitted_datasets(DATASET, others=True)
    print(", ".join([str(len(ds)) for ds in datasets]))
