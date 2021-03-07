from pathlib import Path
from torch.utils.data import TensorDataset
import pandas as pd
import torch

raw_path = Path("~/proben1").expanduser()
pt_path = Path("./datasets/proben1")


def create_thyroid():
    with open(raw_path / "thyroid" / "ann-train.data", "r") as data_file:
        train_data: pd.DataFrame = pd.read_csv(data_file, sep=" ", header=None)

    train_data = train_data.dropna(axis=1)
    x_train, y_train = torch.Tensor(train_data.iloc[:, :21].to_numpy()), torch.Tensor(train_data.iloc[:, 21].to_numpy())
    del train_data

    print(x_train.size())
    print(y_train.size())

    with open(raw_path / "thyroid" / "ann-test.data", "r") as data_file:
        test_data: pd.DataFrame = pd.read_csv(data_file, sep=" ", header=None)
    
    test_data = test_data.dropna(axis=1)
    x_test, y_test = torch.Tensor(test_data.iloc[:, :21].to_numpy()), torch.Tensor(test_data.iloc[:, 21].to_numpy())
    del test_data

    print(x_test.size())
    print(y_test.size())

    torch.save(TensorDataset(x_train, y_train), pt_path / "thyroid-train.pt")
    torch.save(TensorDataset(x_test, y_test), pt_path / "thyroid-test.pt")


if __name__ == "__main__":
    create_thyroid()
