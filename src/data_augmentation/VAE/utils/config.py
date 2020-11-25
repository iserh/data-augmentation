"""Accessing the config."""
from pathlib import Path

from yaml import SafeLoader, load


class Config:
    """Static Configuration class for project."""

    def __init__(self) -> None:
        """Initialize config class."""
        self.load_config()

    @property
    def model_path(self) -> Path:
        """Gets the path to the directory where trained models are stored.

        Returns:
            Path to model directory
        """
        return Path(self.config["model_path"])

    @property
    def eval_path(self) -> Path:
        """Gets the path to the directory where evaluation results are stored.

        Returns:
            Path to evaluation directory
        """
        return Path(self.config["eval_path"])

    def load_config(self) -> dict:
        """Load the config file.

        Returns:
            Config as a dictionary
        """
        with open("./config.yml", "r") as infile:
            self.config = dict(load(infile, Loader=SafeLoader))
        return self.config


config = Config()

if __name__ == "__main__":
    print(f"Model directory in {config.model_path}")
    print(f"Evaluation directory in {config.eval_path}")
