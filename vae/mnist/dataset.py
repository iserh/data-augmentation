from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MNIST


def MNISTWrapper(train: bool = False) -> MNIST:
    # Load dataset
    mnist = MNIST(
        root="~/torch_datasets",
        transform=transforms.ToTensor(),
        train=train,
        target_transform=lambda t: Tensor([t]),
    )
    print(f"MNISt size: {len(mnist)}")
    return mnist
