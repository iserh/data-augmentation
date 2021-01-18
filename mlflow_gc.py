import os

from utils.config import mlflow_roots

for root in mlflow_roots.values():
    print(f"{root=}")
    os.system(f"mlflow gc --backend-store-uri {root}")
