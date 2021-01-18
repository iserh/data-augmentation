from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA


def CelebALoader(
    batch_size: int,
    shuffle: bool,
    train: bool = False,
    target_type: str = "identity",
    pin_memory: bool = False,
) -> DataLoader:
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
    )
    print(f"Dataset size: {len(celeba)}, {batch_size=}, {shuffle=}")
    return DataLoader(
        celeba,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=pin_memory,
    )
