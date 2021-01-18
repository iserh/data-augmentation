"""Mlflow utilities."""
from pathlib import Path
from urllib.parse import unquote, urlparse

from mlflow.entities import Run


def get_artifact_path(run: Run) -> Path:
    """Get the artifact path from a mlflow run.

    Args:
        run (Run): Mlflow run

    Returns:
        Path: artifact path
    """
    return Path(unquote(urlparse(run.info.artifact_uri).path))
