from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def MNISTLoader(batch_size: int, shuffle: bool, train: bool = False, pin_memory: bool = False) -> DataLoader:
    # Load dataset
    mnist = MNIST(
        root="~/torch_datasets",
        transform=transforms.ToTensor(),
        train=train,
    )
    print(f"Dataset size: {len(mnist)}, {batch_size=}, {shuffle=}")
    return DataLoader(
        mnist,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=pin_memory,
    )
