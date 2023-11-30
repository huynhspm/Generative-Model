from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import pyrootutils
from torch import Tensor
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.dataset import init_dataset


class TransformDataset(Dataset):

    def __init__(self, dataset: Dataset, transform: Optional[Compose] = None):
        self.dataset = dataset

        assert transform is not None, ('transform is None')
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, cond = self.dataset[idx]
        if cond is not None and 'image' in cond.keys():
            transformed = self.transform(image=image, cond=cond['image'])
            image, cond['image'] = transformed["image"], transformed["cond"]
        else:
            image = self.transform(image=np.array(image))["image"]

        return image, cond


class DiffusionDataModule(pl.LightningDataModule):
    """
    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "./data",
        train_val_test_split: Tuple[float, float, float]
        | Tuple[int, int, int] = (0.8, 0.1, 0.1),
        transform_train: Optional[Compose] = None,
        transform_val: Optional[Compose] = None,
        batch_size: int = 64,
        num_workers: int = 2,
        pin_memory: bool = False,
        dataset_name: str = 'mnist',
        n_classes: str = 10,
    ):
        """
        data_dir: 
        train_val_test_split: 
        transform_train:
        transform_val:
        batch_size:
        num_workers:
        pin_memory:
        dataset_name:
        n_classes:
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return self.hparams.n_classes

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = init_dataset(self.hparams.dataset_name,
                                   data_dir=self.hparams.data_dir)

            print('Dataset:', len(dataset))
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            print('Train-Val-Test:', len(self.data_train), len(self.data_val),
                  len(self.data_test))
            self.data_train = TransformDataset(
                dataset=self.data_train,
                transform=self.hparams.transform_train)
            self.data_val = TransformDataset(
                dataset=self.data_val, transform=self.hparams.transform_val)
            self.data_test = TransformDataset(
                dataset=self.data_test, transform=self.hparams.transform_val)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "data")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="sketch_celeba.yaml")
    def main(cfg: DictConfig):
        print(cfg)

        datamodule: DiffusionDataModule = hydra.utils.instantiate(
            cfg, data_dir=f"{root}/data")
        datamodule.setup()

        train_dataloader = datamodule.train_dataloader()
        print('train_dataloader:', len(train_dataloader))

        batch_image = next(iter(train_dataloader))
        image, cond = batch_image
        cond_label = cond['label'] if 'label' in cond.keys() else None
        cond_image = cond['image'] if 'image' in cond.keys() else None

        print('Image shape:', image.shape)

        if cond_label is not None:
            print('Cond label shape:', cond_label.shape)

        if cond_image is not None:
            print('Cond image shape:', cond_image.shape)

        visualize(image, cond_image, cond_label)

    def visualize(image: Tensor, cond_image: Tensor | None,
                  cond_label: Tensor | None):
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid, save_image

        mean = 0.5
        std = 0.5
        image = ((image * std + mean))
        image = make_grid(image[:25], nrow=5)

        if cond_label is not None:
            print(cond_label[:25])

        if cond_image is None:
            save_image(image, 'image.jpg')
            plt.imshow(image.moveaxis(0, 2))
            plt.show()
        else:
            cond_image = ((cond_image * std + mean))
            cond_image = make_grid(cond_image[:25], nrow=5)

            save_image(image, 'image.jpg')
            save_image(cond_image, 'cond.jpg')

            plt.figure(figsize=(16, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(image.moveaxis(0, 2))
            plt.title('Image')
            plt.subplot(1, 2, 2)
            plt.imshow(cond_image.moveaxis(0, 2))
            plt.title('Condition')
            plt.show()

    main()