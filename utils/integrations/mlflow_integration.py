"""Mlflow Session."""
from enum import Enum
from typing import Any, Dict, Union

import mlflow
from mlflow.entities.run import Run
from torch.nn.modules import Module

from utils.models import PretrainedConfig


class BackendStore(Enum):
    """Enum containing backend-store-uri's for mlflow."""

    Default = "experiments/Default"
    MNIST = "experiments/MNIST"
    CelebA = "experiments/CelebA"


class ExperimentName(Enum):
    """Enum containing backend-store-uri's for mlflow."""

    Default = "Default"
    VAETrain = "VAE Training"
    MLP = "MLP"
    VAEGeneration = "VAE Generation"
    FeatureSpace = "Feature Space"
    TEST = "_TEST_"


def get_run(experiment_name: Union[ExperimentName, str], **params: Dict[str, Any]) -> Run:
    # convert experiment name to 'str' if it isn't already 'str'
    experiment_name = experiment_name.value if isinstance(experiment_name, ExperimentName) else experiment_name
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


def load_pytorch_model(experiment_name: Union[ExperimentName, str], config: PretrainedConfig) -> Module:
    # convert config to dict and remove checkpoint key (cause it's no mlflow param)
    run_params = dict(config.__dict__)
    del run_params["checkpoint"]
    del run_params["compute_loss"]
    # get the run
    run = get_run(experiment_name, **run_params)
    # load the model and return it
    return mlflow.pytorch.load_model(
        run.artifact_uri + f"/models/model-epoch={config.checkpoint}",
    )
