from typing import Any, Dict, Optional, Tuple

import sys

sys.path.insert(0, '')

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from src.datamodule.dataset import init_dataset


class DiffusionDataModule(LightningDataModule):
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
        root: str = "./data",
        name: str = None,
        img_dims: Tuple[int, int, int] = [1, 32, 32],
        augmentation=False,
        batch_size: int = 64,
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        """
        root: 
        name: name of dataset
        img_dims: the resolution of image
        augmentation: 
        batch_size:
        num_workers:
        pin_memory:
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

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
            data_full = init_dataset(
                self.hparams.name,
                root=self.hparams.root,
                img_dims=self.hparams.img_dims,
                augmentation=self.hparams.augmentation,
            )

            full_length = len(data_full)
            val_length = int(full_length * 0.1)
            test_length = val_length
            train_length = full_length - val_length - test_length

            train_val_test_split = [train_length, val_length, test_length]

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=data_full,
                lengths=train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

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
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" /
                                   "diffusion_datamodule.yaml")

    datamodule = hydra.utils.instantiate(cfg)

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()

    import torch
    import matplotlib.pyplot as plt

    batch_image = next(iter(train_dataloader))
    image = batch_image[0]

    image = (image.clamp(-1, 1) + 1) / 2
    image = (image * 255).type(torch.uint8)

    plt.imshow(image.moveaxis(0, 2))
    plt.show()