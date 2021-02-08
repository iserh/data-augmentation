# """Variational autoencoder module class."""
# from typing import Tuple

# import torch.nn as nn
# from torch import Tensor

# from utils import init_weights

# from .base_model import DecoderBaseModel, EncoderBaseModel, VAEModel, VAEConfig, load_pretrained_model


# class CelebAEncoder(EncoderBaseModel):
#     def __init__(self, z_dim: int, nc: int) -> None:
#         super().__init__(z_dim, nc)
#         self.conv_stage = nn.Sequential(
#             # input is (n_channels) x 64 x 64
#             nn.Conv2d(nc, 64, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (64) x 32 x 32
#             nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (64*2) x 16 x 16
#             nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (64*4) x 8 x 8
#             nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (64*8) x 4 x 4
#             nn.Flatten(),
#         )
#         # Encoder mean
#         self.mean = nn.Linear(64 * 8 * 4 * 4, z_dim)
#         # Encoder Variance log
#         self.variance_log = nn.Linear(64 * 8 * 4 * 4, z_dim)

#         # initialize weights
#         self.conv_stage.apply(init_weights)
#         self.mean.apply(init_weights)
#         self.variance_log.apply(init_weights)

#     def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
#         x = self.conv_stage(x)
#         return self.mean(x), self.variance_log(x)


# class CelebADecoder(DecoderBaseModel):
#     def __init__(self, z_dim: int, nc: int) -> None:
#         super().__init__(z_dim, nc)
#         self.linear_stage = nn.Linear(z_dim, 64 * 8 * 4 * 4)
#         self.conv_stage = nn.Sequential(
#             # input is (64*8) x 4 x 4
#             nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 4),
#             nn.ReLU(True),
#             # state size. (64*4) x 8 x 8
#             nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 2),
#             nn.ReLU(True),
#             # state size. (64*2) x 16 x 16
#             nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             # state size. (64) x 32 x 32
#             nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False),
#             # state size. (n_channels) x 64 x 64
#             nn.Sigmoid(),
#         )

#         # initialize weights
#         self.linear_stage.apply(init_weights)
#         self.conv_stage.apply(init_weights)

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.linear_stage(x)
#         x = x.view(x.size(0), 64 * 8, 4, 4)
#         return self.conv_stage(x)


# class CelebAVAE(VAEModel):
#     def __init__(self, z_dim: int) -> None:
#         super().__init__(z_dim)
#         self.code_paths.append(__file__)
#         self.encoder = CelebAEncoder(z_dim, 3)
#         self.decoder = CelebADecoder(z_dim, 3)

#     @staticmethod
#     def from_pretrained(config: VAEConfig) -> "CelebAVAE":
#         return load_pretrained_model(config)
