from typing import Tuple
import torch
import os.path as osp
from torchvision.transforms import (
    Compose,
    ToTensor,
    RandomHorizontalFlip,
    Pad,
)
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import MNIST


class MnistDataset(Dataset):

    dataset_dir = 'mnist'

    def __init__(self,
                 root: str = '',
                 img_dims: Tuple[int, int, int] = [1, 28, 28],
                 augmentation: bool = False) -> None:
        """
            root:
            img_dims:
            augmentation:
        """

        super().__init__()

        transforms = [ToTensor(), Pad(padding=2)]

        if augmentation:
            transforms += [RandomHorizontalFlip(p=0.5)]

        self.transforms = Compose(transforms)
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.prepare_data()

    def prepare_data(self) -> None:
        trainset = MNIST(self.dataset_dir,
                         download=True,
                         train=True,
                         transform=self.transforms)

        testset = MNIST(self.dataset_dir,
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