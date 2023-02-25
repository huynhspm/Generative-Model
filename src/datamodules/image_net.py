import glob
import cv2
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Tuple


class ImageNetDataset(Dataset):

    def __init__(self, data_dir: str, size: Tuple[int, int],
                 augmentation: bool) -> None:
        super().__init__()
        if augmentation:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
            ])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

        img_dir = [
            f"{data_dir}/test/images/*.JPEG",
            f"{data_dir}/train/*/images/*.JPEG",
            f"{data_dir}/val/images/*.JPEG",
        ]

        self.img_paths = glob.glob(img_dir[0]) + glob.glob(
            img_dir[1]) + glob.glob(img_dir[2])

        # self.prepare_data()

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image)
        return image * 2.0 - 1.0


class ImageNetDataModule(LightningDataModule):

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
        data_full = ImageNetDataset(data_dir=f"{self.data_dir}",
                                    size=self.dims[1:3],
                                    augmentation=self.augmentation)

        print(len(data_full))
        self.train_dataset, self.val_dataset = random_split(
            data_full, [100000, 20000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


if __name__ == "__main__":
    dm = ImageNetDataModule(16, 2, [3, 64, 64], "./data/tiny-imagenet", True)
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
