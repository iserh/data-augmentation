"""Containins functions, class for applying VAEs."""
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset

from utils.data import BatchCollector, LambdaDataset, LoaderDataset
from vae.model import VariationalAutoencoder


class AppliedVAE:
    def __init__(self, model: VariationalAutoencoder, cuda: bool = True) -> None:
        self._load_model(model, cuda)

    def encode_dataset(self, dataset: Dataset, shuffle: bool = False) -> LoaderDataset:
        """Decodes a dataset."""
        raw_loader = DataLoader(dataset, batch_size=512, num_workers=4, shuffle=shuffle)
        raw_set_batched = LoaderDataset(raw_loader)
        encoded_set = LambdaDataset(raw_set_batched, BatchCollector(self._encode))
        encoded_loader = DataLoader(encoded_set, batch_size=512, collate_fn=BatchCollector.collate_fn)
        return LoaderDataset(encoded_loader)

    def decode_dataset(
        self,
        dataset: Union[LoaderDataset, TensorDataset],
    ) -> LoaderDataset:
        """Decodes a latent dataset."""
        decoded_set = LambdaDataset(dataset, BatchCollector(self._decode))
        decoded_loader = DataLoader(decoded_set, batch_size=512, collate_fn=BatchCollector.collate_fn)
        return LoaderDataset(decoded_loader)

    @torch.no_grad()
    def _encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode an input tensor."""
        m, log_v = self.vae.encoder(x.to(self.device))
        return m.cpu(), log_v.cpu(), y

    @torch.no_grad()
    def _decode(self, z: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode a latent tensor."""
        return self.vae.decoder(z.to(self.device)).cpu(), y

    # *** private functions ***

    def _load_model(self, model: VariationalAutoencoder, cuda: bool = True) -> None:
        # Use cuda if available
        self.device = "cuda:0" if cuda and torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)
        # Move model to device
        self.vae = model.to(self.device)
        self.vae.eval()


if __name__ == "__main__":
    from utils.mlflow_utils import ExperimentTypes, Session, get_run, load_pytorch_model
    from vae.celeba.dataset import CelebAWrapper
    from vae.mnist.dataset import MNISTWrapper

    DATASET = "CelebA"
    vae_hparams = {
        "EPOCHS": 240,
        "Z_DIM": 128,
        "BETA": 1.0,
    }

    Session.set_backend_root(root=DATASET)

    model = load_pytorch_model(get_run(ExperimentTypes.VAETraining, **vae_hparams), chkpt=vae_hparams["EPOCHS"])
    vae = AppliedVAE(model, cuda=True)
    dataset = CelebAWrapper(train=False)

    encoded_dataset = vae.encode_dataset(dataset, shuffle=True)
    decoded_dataset = vae.decode_dataset(encoded_dataset)
    decode_loader = DataLoader(decoded_dataset, batch_size=512, collate_fn=BatchCollector.collate_fn)

    print(next(iter(decode_loader))[0].size())
