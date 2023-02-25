from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from typing import Tuple


class MnistDataset(Dataset):

    def __init__(self, train: bool, data_dir: str, augmentation: bool) -> None:
        super().__init__()
        self.train = train
        self.data_dir = data_dir

        if augmentation:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Pad(2)])

        self.prepare_data()

    def prepare_data(self) -> None:
        self.dataset = MNIST(self.data_dir,
                             download=True,
                             train=self.train,
                             transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        image = image * 2.0 - 1.0
        return image


class MnistDataModule(LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int, dims: Tuple[int, int,
                                                                      int],
                 data_dir: str, augmentation: bool) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = dims
        self.data_dir = data_dir
        self.augmentation = augmentation

    def setup(self, stage=None) -> None:
        self.train_dataset = MnistDataset(train=True,
                                          data_dir=self.data_dir,
                                          augmentation=self.augmentation)
        self.val_dataset = MnistDataset(train=False,
                                        data_dir=self.data_dir,
                                        augmentation=self.augmentation)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


if __name__ == "__main__":
    dm = MnistDataModule(128, 2, [1, 32, 32], "./data", True)
    print(dm.dims)
    dm.setup()
    train_dataloader = dm.train_dataloader()

    import torch
    import matplotlib.pyplot as plt

    batch_image = next(iter(train_dataloader))
    image = batch_image[0]

    image = (image.clamp(-1, 1) + 1) / 2
    image = (image * 255).type(torch.uint8)

    plt.imshow(image.moveaxis(0, 2))
    plt.show()
