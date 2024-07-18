from typing import Any, Dict, Optional, Tuple

import torch
import pyrootutils
from torch import Tensor
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from albumentations import Compose

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
            if 'masks' in cond.keys():
                transformed = self.transform(image=image,
                                             cond=cond['image'],
                                             masks=cond['masks'])
                image, cond['image'], cond['masks'] = transformed[
                    "image"], transformed["cond"], transformed['masks']
            else:
                transformed = self.transform(image=image, cond=cond['image'])
                image, cond['image'] = transformed["image"], transformed[
                    "cond"]
        else:
            image = self.transform(image=image)["image"]

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
        train_val_test_dir: Tuple[str, str, str] = None,
        train_val_test_split: Tuple[float, float, float]
        | Tuple[int, int, int] = (0.8, 0.1, 0.1),
        transform_train: Optional[Compose] = None,
        transform_val: Optional[Compose] = None,
        batch_size: int = 64,
        num_workers: int = 2,
        pin_memory: bool = False,
        dataset_name: str = 'mnist',
        n_classes: str = 10,
        image_size: int = 32,
    ) -> None:
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
    def num_classes(self) -> int:
        return self.hparams.n_classes

    @property
    def image_size(self) -> int:
        return self.hparams.image_size

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
            
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.train_val_test_dir:
                train_dir, val_dir, test_dir = self.hparams.train_val_test_dir

                train_set = init_dataset(self.hparams.dataset_name,
                                         data_dir=self.hparams.data_dir,
                                         train_val_test_dir=train_dir)

                val_set = init_dataset(self.hparams.dataset_name,
                                       data_dir=self.hparams.data_dir,
                                       train_val_test_dir=val_dir)

                test_set = init_dataset(self.hparams.dataset_name,
                                        data_dir=self.hparams.data_dir,
                                        train_val_test_dir=test_dir)

            else:
                dataset = init_dataset(self.hparams.dataset_name,
                                       data_dir=self.hparams.data_dir)

                # for testing code before training
                len_dataset = sum(self.hparams.train_val_test_split)
                if 1 < len_dataset and len_dataset < len(dataset):
                    dataset = Subset(dataset, list(range(len_dataset)))

                train_set, val_set, test_set = random_split(
                    dataset=dataset,
                    lengths=self.hparams.train_val_test_split,
                    generator=torch.Generator().manual_seed(42),
                )

            self.data_train = TransformDataset(
                dataset=train_set, transform=self.hparams.transform_train)
            self.data_val = TransformDataset(
                dataset=val_set, transform=self.hparams.transform_val)
            self.data_test = TransformDataset(
                dataset=test_set, transform=self.hparams.transform_val)

            print('Train-Val-Test:', len(self.data_train), len(self.data_val),
                  len(self.data_test))

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
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
                config_name="lidc.yaml")
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
        masks = cond['masks'] if 'masks' in cond.keys() else None
          
        print('Image shape:', image.shape, image.dtype)

        if cond_label is not None:
            print('Cond label shape:', cond_label.shape, cond_label.dtype)

        if cond_image is not None:
            print('Cond image shape:', cond_image.shape, cond_image.dtype)

        visualize(image, cond_image, cond_label, masks, save_img=False)
        # gen_noise_image(xt=cond_image[0])

    def visualize(image: Tensor,
                  cond_image: Tensor | None,
                  cond_label: Tensor | None,
                  masks: Tensor | None,
                  save_img: bool = False):
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid, save_image

        mean = 0.5
        std = 0.5
        image = ((image * std + mean))

        n_image = 9
        n_row = 3

        image = make_grid(image[:n_image], nrow=n_row, pad_value=1)

        if cond_label is not None:
            print(cond_label[:n_image])

        if cond_image is None:
            if save_img:
                save_image(image, 'image.jpg')

            plt.imshow(image.moveaxis(0, 2))
            plt.show()
        else:
            cond_image = ((cond_image * std + mean))
            cond_image = make_grid(cond_image[:n_image],
                                   nrow=n_row,
                                   pad_value=1)
            # brats dataset
            if cond_image.shape[0] == 4:
                if save_img:
                    save_image(image, 'image.jpg')
                    save_image(cond_image[0:1], 't1.jpg')
                    save_image(cond_image[1:2], 't1ce.jpg')
                    save_image(cond_image[2:3], 't2.jpg')
                    save_image(cond_image[3:4], 'flair.jpg')

                    plt.imshow(cond_image.moveaxis(0, 2))
                    plt.axis('off')
                    plt.savefig('cond.jpg')

                plt.figure(figsize=(16, 8))
                plt.subplot(1, 5, 1)
                plt.imshow(cond_image[0:1].moveaxis(0, 2), cmap='gray')
                plt.title('FLAIR')
                plt.subplot(1, 5, 2)
                plt.imshow(cond_image[1:2].moveaxis(0, 2), cmap='gray')
                plt.title('T1')
                plt.subplot(1, 5, 3)
                plt.imshow(cond_image[2:3].moveaxis(0, 2), cmap='gray')
                plt.title('T1CE')
                plt.subplot(1, 5, 4)
                plt.imshow(cond_image[3:4].moveaxis(0, 2), cmap='gray')
                plt.title('T2')
                plt.subplot(1, 5, 5)
                plt.imshow(image.moveaxis(0, 2), cmap='gray')
                plt.title('Mask')
                plt.show()

            # lidc dataset
            elif masks is not None:
                masks = ((masks * std + mean))
                masks = make_grid(masks[:n_image], nrow=n_row, pad_value=1)

                if save_img:
                    save_image(image, 'image.jpg')
                    save_image(cond_image, 'cond.jpg')
                    save_image(masks[0:1], 'mask0.jpg')
                    save_image(masks[1:2], 'mask1.jpg')
                    save_image(masks[2:3], 'mask2.jpg')
                    save_image(masks[3:4], 'mask3.jpg')

                plt.figure(figsize=(16, 8))
                plt.subplot(2, 3, 1)
                plt.imshow(image.moveaxis(0, 2))
                plt.title('Image')
                plt.subplot(2, 3, 2)
                plt.imshow(cond_image.moveaxis(0, 2))
                plt.title('Condition')
                plt.subplot(2, 3, 3)
                plt.imshow(masks[0:1].moveaxis(0, 2))
                plt.title('Mask_0')
                plt.subplot(2, 3, 4)
                plt.imshow(masks[1:2].moveaxis(0, 2))
                plt.title('Mask_1')
                plt.subplot(2, 3, 5)
                plt.imshow(masks[2:3].moveaxis(0, 2))
                plt.title('Mask_2')
                plt.subplot(2, 3, 6)
                plt.imshow(masks[3:4].moveaxis(0, 2))
                plt.title('Mask_3')
                plt.show()

            else:
                if save_img:
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

    def gen_noise_image(xt):

        import matplotlib.pyplot as plt
        beta_start = 1e-4
        beta_end = 0.02
        n_steps = 1000
        betas = torch.linspace(beta_start, beta_end, n_steps)

        from torchvision.utils import save_image
        for t in range(n_steps):
            if t % 50 == 0:
                plt.imshow(((xt * 0.5) + 0.5).clamp(0, 1).moveaxis(0, 2))
                save_image(((xt * 0.5) + 0.5).clamp(0, 1), f'{t}.jpg')
                plt.title('Image')
                plt.show()

            noise = torch.randn_like(xt)
            xt = torch.sqrt(1 - betas[t]) * xt + torch.sqrt(betas[t]) * noise

    main()