from typing import List, Any, Tuple, Optional

import torch
import pytorch_lightning as pl
from torchmetrics.image import fid, inception
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback

from src.models import ConditionDiffusionModule


class DiffusionCallback(Callback):

    def __init__(self, gen_shape: Tuple[int, int], gen_type: str,
                 mean: List[float], std: List[float], callbacks: List[str],
                 feature: Optional[int]):
        self.gen_shape = gen_shape
        self.gen_type = gen_type
        self.mean = torch.Tensor(mean).reshape(1, -1, 1, 1)
        self.std = torch.Tensor(std).reshape(1, -1, 1, 1)
        self.callbacks = callbacks
        self.images = {
            'train': None,
            'val': None,
            'test': None,
        }
        self.labels = {
            'train': None,
            'val': None,
            'test': None,
        }

        if 'fid' in callbacks:
            self.fid_metric = fid.FrechetInceptionDistance(feature=feature,
                                                           normalize=True)

        if 'is' in callbacks:
            self.is_metric = inception.InceptionScore(feature=feature,
                                                      normalize=True)

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

            cond = None
            if isinstance(pl_module, ConditionDiffusionModule):
                cond = self.labels[mode][:self.gen_shape[0] * self.gen_shape[1]]

            gen_samples = pl_module.net.get_p_sample(
                num_sample=self.gen_shape[0] * self.gen_shape[1],
                gen_type=self.gen_type,
                device=pl_module.device,
                cond=cond)

            fake = gen_samples[-1]
            fake = (fake * self.std + self.mean).clamp(0, 1)

            real = self.images[mode][:self.gen_shape[0] * self.gen_shape[1]]
            real = (real * self.std + self.mean).clamp(0, 1)

            if 'sample' in self.callbacks:
                self.sample(fake.detach().cpu(),
                            real.detach().cpu(),
                            trainer,
                            nrow=self.gen_shape[0],
                            mode=mode)

            if 'fid' in self.callbacks:
                self.compute_fid(fake, real, pl_module, mode=mode)

            if 'is' in self.callbacks:
                self.compute_is(fake, pl_module, mode=mode)

            self.images[mode] = None
            self.labels[mode] = None

    def sample(self, fake: torch.Tensor, real: torch.Tensor,
               trainer: pl.Trainer, nrow: int, mode: str):
        fake = make_grid(fake, nrow=nrow)
        real = make_grid(real, nrow=nrow)

        # logging
        trainer.logger.log_image(key=mode + '/sample',
                                 images=[fake, real],
                                 caption=["fake", "real"])

    def compute_fid(self, fake: torch.Tensor, real: torch.Tensor,
                    pl_module: pl.LightningDataModule, mode: str):
        # reset
        self.fid_metric.reset()
        self.fid_metric.to(pl_module.device)

        # update
        self.fid_metric.update(fake, real=False)
        self.fid_metric.update(real, real=True)

        # logging
        pl_module.log(mode + '/fid',
                      self.fid_metric.compute(),
                      prog_bar=True,
                      sync_dist=True)

    def compute_is(self, fake: torch.Tensor, pl_module: pl.LightningDataModule,
                   mode: str):
        # reset
        self.is_metric.reset()
        self.is_metric.to(pl_module.device)

        # update
        self.is_metric.update(fake)

        #logging
        mean, std = self.is_metric.compute()
        range = {
            'min': mean - std,
            'max': mean + std,
        }

        pl_module.log(mode + '/is', range, prog_bar=True, sync_dist=True)

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
            self.images[mode], self.labels[mode] = batch
        elif self.images[mode].shape[0] < self.gen_shape[0] * self.gen_shape[1]:
            self.images[mode] = torch.cat((self.images[mode], batch[0]), dim=0)
            self.labels[mode] = torch.cat((self.labels[mode], batch[1]), dim=0)