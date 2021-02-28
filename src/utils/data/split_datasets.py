from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data.dataset import TensorDataset
from yaml.loader import SafeLoader
from utils.data import get_dataset
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import yaml

data_path = "datasets/splitted"


def split_datasets(dataset_name: str, reduce: Optional[int] = None, seed: Optional[int] = None) -> None:
    # seed
    if seed is not None:
        torch.manual_seed(seed)
    # load dataset
    dataset = get_dataset(DATASET, train=True)
    # optionally reduce size of the dataset
    if reduce is not None:
        dataset = Subset(dataset, torch.randperm(len(dataset))[:500])
    # extract data from the dataset
    inputs, labels = next(iter(DataLoader(dataset, batch_size=len(dataset))))
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
    
    # save the train dataset unsplitted
    torch.save(dataset, path / "train.pt")

    # iterate over classes
    for label in classes.tolist():
        mask_class = labels == label
        mask_other = labels != label
        labels_class = labels[mask_class]
        labels_other = labels[mask_other]
        inputs_class = inputs[mask_class]
        inputs_other = inputs[mask_other]

        # select amount of others to reach a 20% part in total
        n_others = len(labels_class) // 4
        others_idx = torch.randperm(len(labels_other))[:n_others]

        # combine all class inputs/labels with the 20% part of other classes
        train_inputs_class = torch.cat([inputs_class, inputs_other[others_idx]])
        train_labels_class = torch.cat([labels_class, labels_other[others_idx]])
        # save this dataset
        torch.save(TensorDataset(train_inputs_class, train_labels_class), path / "train" / f"class-{label}.pt")


def load_splitted_datasets(dataset_name: str) -> Tuple[List[TensorDataset], Dict[str, Any]]:
    path = Path(data_path) / dataset_name
    if not path.exists():
        raise FileNotFoundError("Dataset does not exist.")

    with open(path / "info.yml", "r") as yml_file:
        info = yaml.load(yml_file, Loader=SafeLoader)

    return [torch.load(path / "train" / f"class-{label}.pt") for label in info["classes"]], info


if __name__ == "__main__":
    DATASET = "MNIST"
    # split_datasets(DATASET, reduce=200, seed=1337)
    datasets, _ = load_splitted_datasets(DATASET)
    print(", ".join([str(len(ds)) for ds in datasets]))
