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


class DogCatDataset(Dataset):

    dataset_dir = 'dogcat'
    dataset_url = 'https://www.kaggle.com/datasets/tongpython/cat-and-dog'

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
            f"{self.dataset_dir}/test_set/cats/*.jpg",
            f"{self.dataset_dir}/test_set/dogs/*.jpg",
            f"{self.dataset_dir}/training_set/cats/*.jpg",
            f"{self.dataset_dir}/training_set/dogs/*.jpg",
            f"{self.dataset_dir}/train/cat/*.jpg",
            f"{self.dataset_dir}/train/dog/*.jpg",
            f"{self.dataset_dir}/train/wild/*.jpg",
            f"{self.dataset_dir}/val/cat/*.jpg",
            f"{self.dataset_dir}/val/dog/*.jpg",
            f"{self.dataset_dir}/val/wild/*.jpg",
        ]

        self.img_paths = glob.glob(img_dir[0]) + glob.glob(
            img_dir[1]) + glob.glob(img_dir[2]) + glob.glob(
                img_dir[3]) + glob.glob(img_dir[4]) + glob.glob(
                    img_dir[5]) + glob.glob(img_dir[6]) + glob.glob(img_dir[7])

    # self.prepare_data()

    def prepare_data(self) -> None:
        import opendatasets as od
        od.download(self.dataset_url)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index) -> torch.Tensor:
        img_path = self.img_paths[index]
        image = imageio.imread(img_path)

        if self.transforms:
            image = self.transforms(image)
        return image * 2.0 - 1.0
