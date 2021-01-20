import torch
from torchvision import transforms
from torchvision.datasets import CelebA


def CelebAWrapper(train: bool = False, target_type: str = "identity") -> CelebA:
    # Load dataset
    celeba = CelebA(
        root="~/torch_datasets",
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        ),
        split="train" if train else "test",
        target_type=target_type,
        target_transform=torch.as_tensor,
    )
    print(f"CelebA size: {len(celeba)}")
    return celeba
