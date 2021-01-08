"""Utils."""
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from mlflow.entities import Run


def get_artifact_path(run: Run) -> Path:
    """Get the artifact path from a mlflow run.

    Args:
        run (Run): Mlflow run

    Returns:
        Path: artifact path
    """
    return Path(unquote(urlparse(run.info.artifact_uri).path))


def reshape_to_img(
    x: np.ndarray, width: int, height: int, rows: int, cols: int
) -> np.ndarray:
    """Reshape numpy array to work with plt.imshow().

    Args:
        x (np.ndarray): Input array
        width (int): Width of each image
        height (int): Height of each image
        rows (int): Number of rows
        cols (int): Number of columns

    Returns:
        np.ndarray: Reshaped array
    """
    return (
        x.reshape(rows, cols, height, width)
        .transpose(0, 2, 1, 3)
        .reshape(rows * height, cols * width)
    )
