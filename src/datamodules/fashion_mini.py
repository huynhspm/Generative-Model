from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from typing import Tuple
import torch


class FashionMiniDataModule(LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, dims: Tuple[int, int, int], data_dir: str) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.dims = dims

    def setup(self, stage=None) -> None:
        train_dataset = FashionMNIST(self.data_dir, train=True)
        val_dataset = FashionMNIST(self.data_dir, train=False)

        train_dataset = torch.Tensor(train_dataset.data)[0: 12000]
        val_dataset = torch.Tensor(val_dataset.data)[0: 2000]

        pad = transforms.Pad(2)
        train_dataset = pad(train_dataset)
        val_dataset = pad(val_dataset)

        train_dataset = train_dataset.unsqueeze(1)
        val_dataset = val_dataset.unsqueeze(1)

        self.train_dataset = ((train_dataset / 255.0) * 2.0) - 1.0
        self.val_dataset = ((val_dataset / 255.0) * 2.0) - 1.0

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    dm = FashionMiniDataModule(128, 2, [1, 32, 32], "./data")
    dm.setup()
