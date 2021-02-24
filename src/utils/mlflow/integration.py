"""Mlflow integration."""
import os
from collections import namedtuple

backend_stores = namedtuple("BackendStores", ["Default", "MNIST", "CIFAR10", "Omniglot", "KMNIST"])(
    Default="experiments/Default",
    MNIST="experiments/MNIST",
    CIFAR10="experiments/CIFAR10",
    Omniglot="experiments/Omniglot",
    KMNIST="experiments/KMNIST",
)


def run_garbage_collection(*paths: str) -> None:
    """Run mlflow garbage collection of backend store paths.

    Args:
        *paths (str): Paths to run garbage collection on
    """
    for path in paths:
        print(f"Running garbage collection on {path}")
        os.system(f"mlflow gc --backend-store-uri {path}")


if __name__ == "__main__":
    run_garbage_collection(*list(backend_stores))
