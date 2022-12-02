from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, Dataset
from typing import Tuple


class FashionDataset(Dataset):
    def __init__(self, train: bool, data_dir: str) -> None:
        super().__init__()
        self.train = train
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Pad(2)])
        self.prepare_data()

    def prepare_data(self) -> None:
        self.dataset = FashionMNIST(self.data_dir, download=True,
                                    train=self.train, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        image = image * 2.0 - 1.0
        return image


class FashionDataModule(LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, dims: Tuple[int, int, int], data_dir: str) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = dims
        self.data_dir = data_dir

    def setup(self, stage=None) -> None:
        self.train_dataset = FashionDataset(train=True, data_dir=self.data_dir)
        self.val_dataset = FashionDataset(train=False, data_dir=self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    dm = FashionDataModule(128, 2, [1, 32, 32], "./data")
    dm.setup()
