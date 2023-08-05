from typing import List, Any, Tuple

import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback


class AutoEncoderCallback(Callback):

    def __init__(self, grid_shape: Tuple[int, int],
                 mean: List[float], std: List[float]) :
        self.grid_shape = grid_shape
        self.mean = torch.Tensor(mean).reshape(1, -1, 1, 1)
        self.std = torch.Tensor(std).reshape(1, -1, 1, 1)
        self.images = {
            'train': None,
            'val': None,
            'test': None,
        }

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningDataModule):
        self.callback(trainer, pl_module, mode='train')

    def on_validation_epoch_end(self, trainer: pl.Trainer,
                                pl_module: pl.LightningDataModule):
        self.callback(trainer, pl_module, mode='val')

    def on_test_epoch_end(self, trainer: pl.Trainer,
                          pl_module: pl.LightningDataModule):
        self.callback(trainer, pl_module, mode='test')

    def callback(self, trainer: pl.Trainer, pl_module: pl.LightningDataModule,
                 mode: str):
        with torch.no_grad():
            self.mean = self.mean.to(pl_module.device)
            self.std = self.std.to(pl_module.device)

            targets = self.images[mode][:self.grid_shape[0] * self.grid_shape[1]]
            preds = pl_module.net(targets)

            targets = make_grid(targets.cpu(), nrow=self.grid_shape[0])
            preds = make_grid(preds.cpu(), nrow=self.grid_shape[0])

            trainer.logger.log_image(key=mode + '/inference',
                                 images=[preds, targets],
                                 caption=["preds", "targets"])
            
            self.images[mode] = None

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
       self.store_data(batch, mode='train')

    def on_validation_batch_end(self, trainer: pl.Trainer,
                                pl_module: pl.LightningModule, outputs: Any,
                                batch: Any, batch_idx: int,
                                dataloader_idx: int) -> None:
        self.store_data(batch, mode='val')

    def on_test_batch_end(self, trainer: pl.Trainer,
                          pl_module: pl.LightningModule, outputs: Any,
                          batch: Any, batch_idx: int,
                          dataloader_idx: int) -> None:
        self.store_data(batch, mode='test')

    def store_data(self, batch: Any, mode: str):
        if self.images[mode] == None:
            self.images[mode] = batch[0]
        elif self.images[mode].shape[0] < self.grid_shape[0] * self.grid_shape[1]:
            self.images[mode] = torch.cat((self.images[mode], batch[0]), dim=0)