"""Utils."""
from pathlib import Path
from urllib.parse import unquote, urlparse

from mlflow.entities import Run
from PIL.Image import Image
from torchvision.transforms import ToPILImage
from torch import Tensor


def get_artifact_path(run: Run) -> Path:
    """Get the artifact path from a mlflow run.

    Args:
        run (Run): Mlflow run

    Returns:
        Path: artifact path
    """
    return Path(unquote(urlparse(run.info.artifact_uri).path))


class TransformImage:
    """Class for transforming a torch tensor with multiple image tensors to a PIL
    Image."""

    def __init__(self, channels: int, height: int, width: int, mode: str = "L") -> None:
        """Initialize Image transform.

        Args:
            channels (int): Number of image channels
            height (int): Height of each image
            width (int): Width of each image
        """
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.to_pil_img = ToPILImage(mode=mode)

    def __call__(self, x: Tensor, rows: int, cols: int) -> Image:
        """Transform to PIL Image.

        Args:
            x (Tensor): Input tensor of shape [..., channels, height, width]
            rows (int): Number of rows
            cols (int): Number of columns

        Returns:
            Image: PIL Image
        """
        reshaped_img = (
            # (n, channels, height, width) -> (n, height, width, channels)
            x.permute(0, 2, 3, 1)
            .reshape(rows, cols, self.height, self.width, self.channels)
            # (rows, cols, height, width, channels) -> (rows, height, cols, width, channels)
            .permute(0, 2, 1, 3, 4)
            .reshape(rows * self.height, cols * self.width, self.channels)
            # (rows * height, cols * width, channels) -> (channels, rows * height, cols * width)
            .permute(2, 0, 1)
        )
        return self.to_pil_img(reshaped_img)


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
