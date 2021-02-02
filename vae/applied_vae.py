"""Containins functions, class for applying VAEs."""
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset
from tqdm import tqdm

from vae.models import VAEBaseModel


class VAEForDataAugmentation:
    def __init__(self, model: VAEBaseModel, no_cuda: bool = False) -> None:
        self._init_model(model, no_cuda)

    def encode_dataset(self, dataset: Dataset, shuffle: bool = False) -> TensorDataset:
        """Encodes a dataset."""
        # create dataloader for batch computing
        dataloader = DataLoader(dataset, batch_size=512, shuffle=shuffle)
        # encode all batches
        z, _, y = [*zip(*[self._encode(x, y) for x, y in tqdm(dataloader, desc="Encoding")])]
        # return TensorDataset
        return TensorDataset(torch.cat(z, dim=0), torch.cat(y, dim=0))

    def decode_dataset(self, dataset: TensorDataset, shuffle: bool = False) -> TensorDataset:
        """Decodes a dataset with latent vectors."""
        # create dataloader for batch computing
        dataloader = DataLoader(dataset, batch_size=512, shuffle=shuffle)
        # decode all batches
        x_, y = [*zip(*[self._decode(z, y) for z, y in tqdm(dataloader, desc="Decoding")])]
        # return TensorDataset
        return TensorDataset(torch.cat(x_, dim=0), torch.cat(y, dim=0))

    @torch.no_grad()
    def _encode(
        self, x: Tensor, y: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """Encode an input batch."""
        m, log_v = self.vae.encoder(x.to(self.device))
        if y is not None:
            return m.cpu(), log_v.cpu(), y
        else:
            return m.cpu(), log_v.cpu()

    @torch.no_grad()
    def _decode(self, z: Tensor, y: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Decode a latent batch."""
        if y is not None:
            return self.vae.decoder(z.to(self.device)).cpu(), y
        else:
            return self.vae.decoder(z.to(self.device)).cpu()

    # *** private functions ***

    def _init_model(self, model: VAEBaseModel, no_cuda: bool) -> None:
        # Use cuda if available
        self.device = "cuda:0" if torch.cuda.is_available() and not no_cuda else "cpu"
        print("Using device:", self.device)
        # Move model to device
        self.vae = model.to(self.device)
        self.vae.eval()


if __name__ == "__main__":
    import mlflow
    from core.data import MNIST_Dataset

    from utils.integrations import BackendStore
    from vae.models import MNISTVAE, VAEConfig

    DATASET = "MNIST"
    mlflow.set_tracking_uri(BackendStore[DATASET].value)
    vae_config = VAEConfig(epochs=5, checkpoint=5, z_dim=2, beta=1.0)

    model = MNISTVAE.from_pretrained(vae_config)
    vae = VAEForDataAugmentation(model)
    dataset = MNIST_Dataset(train=False)

    encoded_dataset = vae.encode_dataset(dataset, shuffle=True)
    decoded_dataset = vae.decode_dataset(encoded_dataset)
    decode_loader = DataLoader(decoded_dataset, batch_size=512, collate_fn=BatchCollector.collate_fn)

    print(next(iter(decode_loader))[0].size())
