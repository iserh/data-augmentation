"""Mlflow integration."""
import os
from collections import namedtuple
import importlib

backend_stores = namedtuple("BackendStores", ["Default", "MNIST", "CIFAR10", "thyroid", "diabetes", "CelebA"])(
    Default="experiments/Default",
    MNIST="experiments/MNIST",
    CIFAR10="experiments/CIFAR10",
    thyroid="experiments/thyroid",
    diabetes="experiments/diabetes",
    CelebA="experiments/CelebA",
)

mlflow_spec = importlib.util.find_spec("mlflow")
if mlflow_spec is not None:
    import mlflow
    mlflow.set_tracking_uri(backend_stores.Default)
else:
    mlflow = None


def mlflow_available() -> bool:
    return mlflow is not None


def mlflow_active() -> bool:
    return (mlflow is not None) and (mlflow.active_run() is not None)


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
