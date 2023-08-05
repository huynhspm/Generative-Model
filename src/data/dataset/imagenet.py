from typing import Tuple
import torch
import glob
import imageio
import os.path as osp
from torchvision.transforms import Compose
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):

    dataset_dir = 'imagenet'
    dataset_url = ''

    def __init__(self,
                 data_dir: str = 'data',
                 transforms: Compose = None) -> None:
        """
            data_dir:
            transforms:
        """
        super().__init__()

        self.transforms = transforms
        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        img_dir = [
            f"{self.dataset_dir}/test/images/*.JPEG",
            f"{self.dataset_dir}/train/*/images/*.JPEG",
            f"{self.dataset_dir}/val/images/*.JPEG",
        ]

        self.img_paths = glob.glob(img_dir[0]) + glob.glob(
            img_dir[1]) + glob.glob(img_dir[2])

    def prepare_data(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index) -> torch.Tensor:
        # no label
        img_path = self.img_paths[index]
        image = imageio.imread(img_path)

        if self.transforms:
            image = self.transforms(image)
        return image * 2.0 - 1.0
