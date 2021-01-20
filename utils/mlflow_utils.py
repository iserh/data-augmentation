"""Mlflow Session."""
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union
from urllib.parse import unquote, urlparse

import mlflow
from mlflow.entities.experiment import Experiment as MlflowExperiment
from mlflow.entities.run import Run as MlflowRun
from torch.nn.modules import Module


class Roots(Enum):
    """Enum containing backend-store-uri's for mlflow."""

    Default = "experiments/Default"
    MNIST = "experiments/MNIST"
    CelebA = "experiments/CelebA"
    TEST = "experiments/_TEST_"


class ExperimentTypes(Enum):
    """Enum containing backend-store-uri's for mlflow."""

    Default = "Default"
    VAETraining = "VAE Training"
    VAEGeneration = "VAE Generation"
    FeatureSpace = "Feature Space"
    TEST = "_TEST_"


class Run:
    def __init__(self, run_name: str, experiment_id: int) -> None:
        self.run_name = run_name
        self.experiment_id = experiment_id

    def __enter__(self):
        self.run = mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name)
        self.artifact_path = Path(unquote(urlparse(self.run.info.artifact_uri).path))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            mlflow.end_run("FAILED")
        else:
            mlflow.end_run()


class Experiment:
    """A Mlflow Experiment."""

    def __init__(self, experiment_type: Union[ExperimentTypes, str]) -> None:
        self.experiment_type = experiment_type

    @property
    def experiment_type(self) -> ExperimentTypes:
        return self._experiment_type

    @experiment_type.setter
    def experiment_type(self, experiment_type: Union[ExperimentTypes, str]) -> None:
        self._experiment_type = _validate_experiment_type(experiment_type).value
        self._experiment: MlflowExperiment = mlflow.get_experiment_by_name(self._experiment_type)
        if not self._experiment:
            self._experiment: MlflowExperiment = mlflow.get_experiment(mlflow.create_experiment(self._experiment_type))

    def new_run(self, run_name: str) -> Run:
        """Start a new mlflow run.

        Args:
            run_name (str): Name of the run

        Returns:
            Run: Run
        """
        return Run(run_name, self._experiment.experiment_id)


def get_run(experiment_type: Union[ExperimentTypes, str], **hparams: Dict[str, Any]) -> MlflowRun:
    experiment_type = _validate_experiment_type(experiment_type).value
    experiment = mlflow.get_experiment_by_name(experiment_type)
    if experiment is None:
        print("Session not configured!")  # TODO: Better message
    constraints = [f"params.{key} = '{value}'" for key, value in hparams.items()]
    filter_str = " AND ".join(constraints)
    try:
        run = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=filter_str)
        if len(run.index) > 1 and "tags.SELECT" in run:
            run = run.loc[run["tags.SELECT"] == "True"].iloc[0]
        else:
            run = run.iloc[0]
    except IndexError:
        print("No run with specified hparams found!")
        exit(1)
    return run


def load_pytorch_model(run: MlflowRun, chkpt: int) -> Module:
    return mlflow.pytorch.load_model(
        run.artifact_uri + f"/model-epoch={chkpt}",
    )


def _validate_experiment_type(experiment: Union[ExperimentTypes, str]) -> ExperimentTypes:
    # convert experiment to Experiment
    try:
        experiment = experiment if isinstance(experiment, ExperimentTypes) else ExperimentTypes[experiment]
    except KeyError:
        raise KeyError(f"{experiment} is not a valid Experiment.")
    return experiment


def _validate_root(root: Union[Roots, str]) -> Roots:
    # convert root to Root
    try:
        root = root if isinstance(root, Roots) else Roots[root]
    except KeyError:
        raise KeyError(f"{root} is not a valid Root.")
    return root


if __name__ == "__main__":
    Session.set_backend_root(Roots.TEST)
    exp = Experiment(ExperimentTypes.TEST)
    print(exp.experiment_type)
    with exp.new_run("test run") as run:
        print(run.artifact_path)
