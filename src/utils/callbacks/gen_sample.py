from typing import List, Any, Tuple

import torch
from torch import Tensor
from pytorch_lightning import LightningModule, Trainer
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback

from src.models.diffusion import DiffusionModule, ConditionDiffusionModule
from src.models.vae import VAEModule


class GenSample(Callback):

    def __init__(self, grid_shape: Tuple[int, int], mean: float, std: float):
        self.grid_shape = grid_shape
        self.mean = mean
        self.std = std
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

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.sample(trainer, pl_module, mode='train')

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule):
        self.sample(trainer, pl_module, mode='val')

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.sample(trainer, pl_module, mode='test')

    @torch.no_grad()
    def sample(self, trainer: Trainer, pl_module: LightningModule, mode: str):
        n_samples = self.grid_shape[0] * self.grid_shape[1]

        if isinstance(pl_module, VAEModule):
            targets = self.images[mode][:n_samples]

            if pl_module.use_ema:
                with pl_module.ema_scope():
                    preds = pl_module.net(targets)[0]
                    samples = pl_module.net.sample(n_samples=n_samples,
                                                   device=pl_module.device)
            else:
                preds = pl_module.net(targets)[0]
                samples = pl_module.net.sample(n_samples=n_samples,
                                               device=pl_module.device)

            targets = (targets * self.std + self.mean).clamp(0, 1)
            preds = (preds * self.std + self.mean).clamp(0, 1)

            self.log_sample([samples, preds, targets],
                            trainer=trainer,
                            nrow=self.grid_shape[0],
                            mode=mode,
                            caption=['samples', 'recons_img', 'target'])

        elif isinstance(pl_module, DiffusionModule):
            conds = None
            if isinstance(pl_module, ConditionDiffusionModule):
                conds = self.labels[mode][:n_samples]

            if pl_module.use_ema:
                with pl_module.ema_scope():
                    samples = pl_module.net.sample(num_sample=n_samples,
                                                   device=pl_module.device,
                                                   cond=conds)
            else:
                samples = pl_module.net.sample(num_sample=n_samples,
                                               device=pl_module.device,
                                               cond=conds)

            fakes = samples[-1]
            fakes = (fakes * self.std + self.mean).clamp(0, 1)

            reals = self.images[mode][:n_samples]
            reals = (reals * self.std + self.mean).clamp(0, 1)

            self.log_sample(
                [fakes, reals, conds] if conds is not None
                and len(conds.shape) > 2 else [fakes, reals],
                trainer=trainer,
                nrow=self.grid_shape[0],
                mode=mode,
                caption=['fake', 'real', 'cond'] if conds is not None
                and len(conds.shape) > 2 else ['fake', 'real'])

    def log_sample(self,
                   images: Tensor,
                   trainer: Trainer,
                   nrow: int,
                   mode: str,
                   caption=[str, str]):
        images = [make_grid(image, nrow=nrow) for image in images]

        # logging
        trainer.logger.log_image(key=mode + '/inference',
                                 images=images,
                                 caption=caption)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        self.store_data(batch, mode='train')

    def on_validation_batch_end(self, trainer: Trainer,
                                pl_module: LightningModule, outputs: Any,
                                batch: Any, batch_idx: int,
                                dataloader_idx: int) -> None:
        self.store_data(batch, mode='val')

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs: Any, batch: Any, batch_idx: int,
                          dataloader_idx: int) -> None:
        self.store_data(batch, mode='test')

    def store_data(self, batch: Any, mode: str):
        if self.images[mode] == None:
            self.images[mode], self.labels[mode] = batch
        elif self.images[mode].shape[
                0] < self.grid_shape[0] * self.grid_shape[1]:
            self.images[mode] = torch.cat((self.images[mode], batch[0]), dim=0)
            self.labels[mode] = torch.cat((self.labels[mode], batch[1]), dim=0)