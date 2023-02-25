from typing import Tuple
import torch
import os.path as osp
from torchvision.transforms import (
    Compose,
    ToTensor,
    RandomHorizontalFlip,
    Grayscale,
    Resize,
)
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import CIFAR10


class CifarDataset(Dataset):

    dataset_dir = 'cifar'

    def __init__(self,
                 root: str = '',
                 img_dims: Tuple[int, int, int] = [3, 32, 32],
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
        self.prepare_data()

    def prepare_data(self) -> None:
        trainset = CIFAR10(self.dataset_dir,
                           download=True,
                           train=True,
                           transform=self.transforms)

        testset = CIFAR10(self.dataset_dir,
                          download=True,
                          train=False,
                          transform=self.transforms)

        self.dataset = ConcatDataset(datasets=[trainset, testset])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> torch.Tensor:
        image = self.dataset[index][0]
        image = image * 2.0 - 1.0
        return image