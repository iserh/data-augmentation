"""Mlflow Session."""
from typing import Any, Dict
import os

import mlflow
from mlflow.entities.run import Run

from collections import namedtuple

backend_stores = namedtuple("BackendStores", ["Default", "MNIST", "CelebA"])(Default="experiments/Default", MNIST="experiments/MNIST", CelebA="experiments/CelebA")


def get_run(experiment_name: str, **params: Dict[str, Any]) -> Run:
    # get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    assert experiment is not None, f"Experiment {experiment_name} was not found!"
    # constraints for finding the run
    constraints = [f"params.{key} = '{value}'" for key, value in params.items()]
    # create SQL filter string
    filter_str = " AND ".join(constraints)
    # get the run
    try:
        run = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=filter_str)
        # if more than 1 run, return the one with SELECT tag or the first one if not tags are set
        if len(run.index) > 1 and "tags.SELECT" in run:
            run = run.loc[run["tags.SELECT"] == "True"].iloc[0]
        else:
            run = run.iloc[0]
    except IndexError:
        print("No run with specified hparams found!")
        exit(1)
    return run


def run_garbage_collection(*paths: str) -> None:
    for path in paths:
        print(f"Running garbage collection on {path}")
        os.system(f"mlflow gc --backend-store-uri {path}")


if __name__ == "__main__":
    run_garbage_collection(*list(backend_stores))
