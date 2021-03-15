from pathlib import Path
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

raw_path = Path("~/proben1").expanduser()
pt_path = Path("./datasets/proben1")


def create_thyroid():
    with open(raw_path / "thyroid" / "ann-train.data", "r") as data_file:
        train_data: pd.DataFrame = pd.read_csv(data_file, sep=" ", header=None)

    train_data = train_data.dropna(axis=1)
    print(train_data.head())
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(train_data.iloc[:, :21].values)
    train_data.iloc[:, :21] = pd.DataFrame(x_scaled)
    print(train_data.head())
    x_train, y_train = torch.Tensor(train_data.iloc[:, :21].to_numpy()), torch.LongTensor(train_data.iloc[:, 21].to_numpy())
    del train_data

    print(x_train.size())
    print(y_train.size())

    with open(raw_path / "thyroid" / "ann-test.data", "r") as data_file:
        test_data: pd.DataFrame = pd.read_csv(data_file, sep=" ", header=None)
    
    test_data = test_data.dropna(axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(test_data.iloc[:, :21].values)
    test_data.iloc[:, :21] = pd.DataFrame(x_scaled)
    x_test, y_test = torch.Tensor(test_data.iloc[:, :21].to_numpy()), torch.LongTensor(test_data.iloc[:, 21].to_numpy())
    del test_data

    print(x_test.size())
    print(y_test.size())

    torch.save(TensorDataset(x_train, y_train), pt_path / "thyroid-train.pt")
    torch.save(TensorDataset(x_test, y_test), pt_path / "thyroid-test.pt")


def create_diabetes():
    with open(raw_path / "diabetes" / "pima-indians-diabetes.data", "r") as data_file:
        train_data: pd.DataFrame = pd.read_csv(data_file, sep=",", header=None)

    train_data = train_data.dropna(axis=1)
    print(train_data.head())
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(train_data.iloc[:, :8].values)
    train_data.iloc[:, :8] = pd.DataFrame(x_scaled)
    print(train_data.head())
    train, test = train_test_split(train_data.values, test_size=0.2, shuffle=True, stratify=train_data.iloc[:, 8].values, random_state=np.random.RandomState(1337))
    del train_data
    x_train, y_train = torch.Tensor(train[:, :8]), torch.LongTensor(train[:, 8])
    x_test, y_test = torch.Tensor(test[:, :8]), torch.LongTensor(test[:, 8])

    print(x_train.size())
    print(y_train.size())
    print(x_test.size())
    print(y_test.size())

    torch.save(TensorDataset(x_train, y_train), pt_path / "diabetes-train.pt")
    torch.save(TensorDataset(x_test, y_test), pt_path / "diabetes-test.pt")


if __name__ == "__main__":
    create_thyroid()
