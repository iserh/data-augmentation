"""Containins functions, class for applying VAEs."""
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset
from tqdm import tqdm

from .base import VAEModel, load_pretrained_model, VAEConfig
from torch import nn


class VAEForDataAugmentation(VAEModel):
    def __init__(self, config: VAEConfig, no_cuda: bool = False) -> None:
        super().__init__(config)
        self.code_paths.append([__file__])
        self.device = "cuda:0" if torch.cuda.is_available() and not no_cuda else "cpu"
        self.encoder: nn.Module = NotImplemented
        self.decoder: nn.Module = NotImplemented

    def encode_dataset(self, dataset: Dataset, shuffle: bool = False) -> TensorDataset:
        """Encodes a dataset."""
        # create dataloader for batch computing
        dataloader = DataLoader(dataset, batch_size=512, shuffle=shuffle)
        # encode all batches
        z, _, y = [*zip(*[self._encode(x, y) for x, y in tqdm(dataloader, desc="Encoding", leave=False)])]
        # return TensorDataset
        return TensorDataset(torch.cat(z, dim=0), torch.cat(y, dim=0))

    def decode_dataset(self, dataset: TensorDataset, shuffle: bool = False) -> TensorDataset:
        """Decodes a dataset with latent vectors."""
        # create dataloader for batch computing
        dataloader = DataLoader(dataset, batch_size=512, shuffle=shuffle)
        # decode all batches
        x_, y = [*zip(*[self._decode(z, y) for z, y in tqdm(dataloader, desc="Decoding", leave=False)])]
        # return TensorDataset
        return TensorDataset(torch.cat(x_, dim=0), torch.cat(y, dim=0))

    @staticmethod
    def from_pretrained(config: VAEConfig, no_cuda: bool = False) -> "VAEForDataAugmentation":
        model = load_pretrained_model(config)
        vae = VAEForDataAugmentation(config, no_cuda)
        vae.encoder = model.encoder
        vae.decoder = model.decoder
        vae.eval()
        return vae

    # *** private functions ***

    @torch.no_grad()
    def _encode(
        self, x: Tensor, y: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """Encode an input batch."""
        m, log_v = self.encoder(x.to(self.device))
        if y is not None:
            return m.cpu(), log_v.cpu(), y
        else:
            return m.cpu(), log_v.cpu()

    @torch.no_grad()
    def _decode(self, z: Tensor, y: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Decode a latent batch."""
        if y is not None:
            return self.decoder(z.to(self.device)).cpu(), y
        else:
            return self.decoder(z.to(self.device)).cpu()
