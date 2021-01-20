"""Wrapper for running mlflow gc on all backend-store-uri's."""
import os

from utils.mlflow_utils import Roots

for root in Roots:
    print(f"{root.value=}")
    os.system(f"mlflow gc --backend-store-uri {root.value}")
