"""Wrapper for running mlflow gc on all backend-store-uri's."""
import os

from utils.integrations import BackendStore

for root in BackendStore:
    print(f"{root.value=}")
    os.system(f"mlflow gc --backend-store-uri {root.value}")
