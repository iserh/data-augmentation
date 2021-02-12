"""Containins functions, class for applying VAEs."""
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from vae.models.base import VAEConfig, VAEModel


class VAEForDataAugmentation(VAEModel):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__(config)

    def encode_dataset(self, dataset: Dataset, shuffle: bool = False) -> TensorDataset:
        """Encodes a dataset."""
        # create dataloader for batch computing
        dataloader = DataLoader(dataset, batch_size=512, shuffle=shuffle)
        # encode all batches
        z, log_v, y = [*zip(*[self._encode(x, y) for x, y in tqdm(dataloader, desc="Encoding", leave=False)])]
        # return TensorDataset
        return TensorDataset(torch.cat(z, dim=0), torch.cat(log_v, dim=0), torch.cat(y, dim=0))

    def decode_dataset(self, dataset: TensorDataset, shuffle: bool = False) -> TensorDataset:
        """Decodes a dataset with latent vectors."""
        # create dataloader for batch computing
        dataloader = DataLoader(dataset, batch_size=512, shuffle=shuffle)
        lalala = [self._decode(*batch) for batch in tqdm(dataloader, desc="Decoding", leave=False)]
        # decode all batches
        batched_outputs = [*zip(*lalala)]
        # return TensorDataset
        return TensorDataset(*[torch.cat(batch, dim=0) for batch in batched_outputs])

    @staticmethod
    def from_pretrained(config: VAEConfig, epochs: int) -> "VAEForDataAugmentation":
        pretrained_model = VAEModel.from_pretrained(config, epochs)
        vae = VAEForDataAugmentation(pretrained_model.config)
        vae.encoder = pretrained_model.encoder
        vae.decoder = pretrained_model.decoder
        return vae.to(pretrained_model.config.device).eval()

    # *** private functions ***

    @torch.no_grad()
    def _encode(
        self, x: Tensor, y: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """Encode an input batch."""
        m, log_v = self.encoder(x.to(self.config.device))
        if y is not None:
            return m.cpu(), log_v.cpu(), y
        else:
            return m.cpu(), log_v.cpu()

    @torch.no_grad()
    def _decode(self, z: Tensor, y: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Decode a latent batch."""
        if y is not None:
            return self.decoder(z.to(self.config.device)).cpu(), y
        else:
            return self.decoder(z.to(self.config.device)).cpu()
