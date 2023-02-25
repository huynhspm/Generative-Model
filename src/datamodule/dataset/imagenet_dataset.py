from typing import Tuple
import torch
import glob
import imageio
import os.path as osp
from torchvision.transforms import (
    Compose,
    ToTensor,
    RandomHorizontalFlip,
    Grayscale,
    Resize,
)
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):

    dataset_dir = 'imagenet'
    dataset_url = ''

    def __init__(self,
                 root: str = '',
                 img_dims: Tuple[int, int] = [3, 64, 64],
                 augmentation: bool = False) -> None:
        """
            root:
            img_dims:
            augmentation:
        """

        super().__init__()

        transforms = [ToTensor(), Resize(img_dims[1:3])]

        if (img_dims[0] == 1):
            transforms += [Grayscale(num_output_channels=1)]

        if augmentation:
            transforms += [RandomHorizontalFlip(p=0.5)]

        self.transforms = Compose(transforms)

        self.dataset_dir = osp.join(root, self.dataset_dir)
        img_dir = [
            f"{self.dataset_dir}/test/images/*.JPEG",
            f"{self.dataset_dir}/train/*/images/*.JPEG",
            f"{self.dataset_dir}/val/images/*.JPEG",
        ]

        self.img_paths = glob.glob(img_dir[0]) + glob.glob(
            img_dir[1]) + glob.glob(img_dir[2])

    # self.prepare_data()

    def prepare_data(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index) -> torch.Tensor:
        img_path = self.img_paths[index]
        image = imageio.imread(img_path)

        if self.transforms:
            image = self.transforms(image)
        return image * 2.0 - 1.0
