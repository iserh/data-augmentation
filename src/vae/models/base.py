"""Variational autoencoder module base classes."""
import inspect
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch import Tensor
from yaml.loader import SafeLoader

from utils.models import BaseModel, ModelConfig, ModelOutput

from .loss import VAELoss, VAELossOutput

model_store = Path("./pretrained_models/MNIST")


@dataclass
class VAEConfig(ModelConfig):
    z_dim: Optional[int] = None
    beta: Optional[float] = None


@dataclass
class VAEOutput(ModelOutput):
    loss: Optional[VAELossOutput] = None
    mean: Optional[Tensor] = None
    log_var: Optional[Tensor] = None
    z: Optional[Tensor] = None


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z: Tensor) -> Tensor:
        raise NotImplementedError()


class VAEModel(BaseModel):
    def __init__(self, config: VAEConfig) -> None:
        super(VAEModel, self).__init__(config)
        self.criterion = VAELoss(config.beta)
        self.encoder: Encoder = NotImplemented
        self.decoder: Decoder = NotImplemented

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> VAEOutput:
        # encode input tensor
        mean, log_var = self.encoder(x)
        # reparameterization trick
        eps = torch.empty_like(log_var).normal_()
        z = eps * (0.5 * log_var).exp() + mean
        # decode latent tensor
        x_hat = self.decoder(z)
        # compute loss
        loss = self.criterion(x_hat, x, mean, log_var)
        return VAEOutput(
            prediction=x_hat,
            mean=mean,
            log_var=log_var,
            z=z,
            loss=loss,
        )

    @staticmethod
    def from_pretrained(config: VAEConfig, epochs: int) -> "VAEModel":
        return _load_pretrained_model(config, epochs)

    def save(self, epochs: int) -> None:
        _save_model(self, epochs)


def _load_pretrained_model(config: VAEConfig, epochs: int) -> VAEModel:
    # iterate models dir
    for folder in model_store.iterdir():
        # read the config file
        with open(folder / "config.yml", "r") as yml_file:
            model_dict = yaml.load(yml_file, Loader=SafeLoader)
            vae_config = VAEConfig(**model_dict["vae_config"])
        # if config file matches and n_epochs
        if vae_config == config and model_dict["epochs"] == epochs:
            # load model
            model = _load_torch_model_and_return(folder, vae_config)
            # optionally move model to a different device after loading
            return model.to(config.device) if config.device else model

    raise FileNotFoundError("Found no model with specified criteria.")


def _load_torch_model_and_return(model_path: Path, config: VAEConfig) -> VAEModel:
    # load the model's state dict
    state_dict = torch.load(model_path / "state_dict.pt", map_location=config.device)
    # the path to the module sourcefile containing the model class
    module_path = str(model_path / "source.py").replace(".py", "").replace("/", ".")
    # load the source module
    module = __import__(module_path, fromlist=("_get_model_constructor",))
    # create model
    model: VAEModel = module._get_model_constructor()(config)
    # load state_dict into the model
    model.load_state_dict(state_dict)
    # return the model
    return model


def _save_model(model: VAEModel, epochs: int) -> None:
    # create model config
    config = model.config
    # create model folder
    suffix = "_" + datetime.now().strftime("%y%m%d_%H%M%S")
    filename = model_store / suffix
    filename.mkdir(parents=True)
    # save model's state_dict
    torch.save(model.state_dict(), filename / "state_dict.pt")
    # save model config
    with open(filename / "config.yml", "w") as yml_file:
        yaml.dump({"epochs": epochs, "vae_config": config.__dict__}, yml_file)
    # copy source file
    copyfile(inspect.getfile(model.__class__), filename / "source.py")
