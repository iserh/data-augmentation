from typing import Union
from torch.utils.data.dataset import TensorDataset
from torchvision.transforms import ToTensor
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch


class CelebA(TensorDataset):

    def __init__(self, root: Union[Path, str], train: bool = True, targets: bool = False) -> None:
        # TODO: Load dataset from filesystem
        self.root = Path(root)
        fp = root / "img_align_celeba"
        print(len(list(fp.iterdir())))
        trans = ToTensor()

        img_tensor = torch.cat([trans(Image.open(f)) for f in tqdm(list(fp.iterdir())[])], dim=0)
        super().__init__(img_tensor)

    def save(self):
        torch.save(self.tensors, self.root / "img_align_celeba.pt")


    


dataset = CelebA(
    root=Path.home() / "datasets",
    train=True,
    chunk_size=
)

dataset.save()


