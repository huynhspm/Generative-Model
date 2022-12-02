import glob
import cv2
import numpy as np
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Tuple


class GenderDataset(Dataset):
    def __init__(self, data_dir: str, size: int) -> None:
        super().__init__()
        self.size = size
        self.transforms = transforms.Compose([transforms.ToTensor()])

        img_dir = [f"{data_dir}/Test/Female/*.jpg", f"{data_dir}/Test/Male/*.jpg",
                   f"{data_dir}/Validation/Female/*.jpg", f"{data_dir}/Validation/Male/*.jpg",
                   f"{data_dir}/Train/Female/*.jpg", f"{data_dir}/Train/Male/*.jpg"]

        self.img_paths = glob.glob(img_dir[0]) + glob.glob(img_dir[1]) + glob.glob(
            img_dir[2]) + glob.glob(img_dir[3]) + glob.glob(img_dir[4]) + glob.glob(img_dir[5])

        # self.prepare_data()

    def prepare_data(self) -> None:
        import opendatasets as od
        od.download(
            "https://www.kaggle.com/datasets/yasserhessein/gender-dataset")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)

        if self.transforms:
            image = self.transforms(image)

        return image * 2.0 - 1.0


class GenderDataModule(LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, dims: Tuple[int, int, int], data_dir: str) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = dims
        self.data_dir = data_dir

    def setup(self, stage=None) -> None:
        data_full = GenderDataset(f"{self.data_dir}", self.dims[1: 3])

        print(len(data_full))
        self.train_dataset, self.val_dataset = random_split(
            data_full, [34599, 8000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    dm = GenderDataModule(128, 2, [3, 64, 64], "./data/GENDER")
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
