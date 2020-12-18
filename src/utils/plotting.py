"""Generation utilities."""
import numpy as np


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
