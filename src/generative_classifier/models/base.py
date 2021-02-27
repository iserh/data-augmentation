"""Variational autoencoder module class."""
import inspect
from typing import Optional
import torch.nn as nn
from torch import Tensor
from pathlib import Path

from utils.models import BaseModel, ModelConfig, ModelOutput
import yaml
from yaml import SafeLoader
import torch
from shutil import copyfile
from datetime import datetime

model_store = "generative_classifiers/Default"


class GenerativeClassifierModel(BaseModel):

    def __init__(self, config: ModelConfig) -> None:
        super(GenerativeClassifierModel, self).__init__(config)
        self.model: nn.Module = NotImplemented
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> ModelOutput:
        pred: Tensor = self.model(x)
        loss = self.criterion(pred, y.unsqueeze(1).float())
        return ModelOutput(loss=loss, prediction=(pred > 0).long().squeeze())

    @staticmethod
    def from_pretrained(config: ModelConfig, epochs: int) -> "GenerativeClassifierModel":
        return _load_pretrained_model(config, epochs)

    def save(self, epochs: int) -> None:
        _save_model(self, epochs)


def _load_pretrained_model(config: ModelConfig, epochs: int) -> GenerativeClassifierModel:
    # iterate models dir
    for folder in Path(model_store).iterdir():
        # read the config file
        with open(folder / "config.yml", "r") as yml_file:
            model_dict = yaml.load(yml_file, Loader=SafeLoader)
            vae_config = ModelConfig(**model_dict["model_config"])
        # if config file matches and n_epochs
        if vae_config == config and model_dict["epochs"] == epochs:
            # load model
            model = _load_torch_model_and_return(folder, vae_config)
            # optionally move model to a different device after loading
            return model.to(config.device) if config.device else model

    raise FileNotFoundError("Found no model with specified criteria.")


def _load_torch_model_and_return(model_path: Path, config: ModelConfig) -> GenerativeClassifierModel:
    # load the model's state dict
    state_dict = torch.load(model_path / "state_dict.pt", map_location=config.device)
    # the path to the module sourcefile containing the model class
    module_path = str(model_path / "source.py").replace(".py", "").replace("/", ".")
    # load the source module
    module = __import__(module_path, fromlist=("_get_model_constructor",))
    # create model
    model: GenerativeClassifierModel = module._get_model_constructor()(config)
    # load state_dict into the model
    model.load_state_dict(state_dict)
    # return the model
    return model


def _save_model(model: GenerativeClassifierModel, epochs: int) -> None:
    # create model config
    config = model.config
    # create model folder
    suffix = datetime.now().strftime("%y%m%d_%H%M%S")
    filename = Path(model_store) / suffix
    filename.mkdir(parents=True, exist_ok=True)
    # save model's state_dict
    torch.save(model.state_dict(), filename / "state_dict.pt")
    # save model config
    with open(filename / "config.yml", "w") as yml_file:
        yaml.dump({"epochs": epochs, "model_config": config.__dict__}, yml_file)
    # copy source file
    copyfile(inspect.getfile(model.__class__), filename / "source.py")
