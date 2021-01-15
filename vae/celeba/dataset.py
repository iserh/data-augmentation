import torch
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, Resize, Compose

from utils.utils import TransformImage

# Load dataset from torchvision
data = CelebA(
    root="~/torch_datasets",
    transform=Compose([Resize((64, 64)), ToTensor()]),
    split="train",
    target_type="identity",
    download=False,
)
# setup transform for visualization
img_transform = TransformImage(channels=3, width=64, height=64, mode="RGB")

# visualize a few examples
rows, cols = 5, 10
examples = torch.stack([data[i][0] for i in range(rows * cols)], dim=0)
# transform to image
img = img_transform(examples, rows, cols)
# save image
img.save("./examples.png")

print(examples[0].max())
print(examples[0].min())
