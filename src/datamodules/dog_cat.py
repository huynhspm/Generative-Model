import glob
import cv2
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Tuple


class DogCatDataset(Dataset):

    def __init__(self, data_dir: str, size: Tuple[int, int],
                 augmentation: bool) -> None:
        super().__init__()
        if augmentation:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(size)])

        img_dir = [
            f"{data_dir}/test_set/cats/*.jpg",
            f"{data_dir}/test_set/dogs/*.jpg",
            f"{data_dir}/training_set/cats/*.jpg",
            f"{data_dir}/training_set/dogs/*.jpg",
            f"{data_dir}/train/cat/*.jpg",
            f"{data_dir}/train/dog/*.jpg",
            f"{data_dir}/train/wild/*.jpg",
            f"{data_dir}/val/cat/*.jpg",
            f"{data_dir}/val/dog/*.jpg",
            f"{data_dir}/val/wild/*.jpg",
        ]

        self.img_paths = glob.glob(img_dir[0]) + glob.glob(
            img_dir[1]) + glob.glob(img_dir[2]) + glob.glob(
                img_dir[3]) + glob.glob(img_dir[4]) + glob.glob(
                    img_dir[5]) + glob.glob(img_dir[6]) + glob.glob(img_dir[7])

        # self.prepare_data()

    def prepare_data(self) -> None:
        import opendatasets as od
        od.download("https://www.kaggle.com/datasets/tongpython/cat-and-dog")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image)
        return image * 2.0 - 1.0


class DogCatDataModule(LightningDataModule):

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
        data_full = DogCatDataset(data_dir=f"{self.data_dir}",
                                  size=self.dims[1:3],
                                  augmentation=self.augmentation)

        print(len(data_full))
        self.train_dataset, self.val_dataset = random_split(
            data_full, [20158, 5000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


if __name__ == "__main__":
    dm = DogCatDataModule(16, 2, [3, 64, 64], "./data/dog_cat", True)
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
