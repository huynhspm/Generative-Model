from typing import Any, Tuple

import torch
from torch import Tensor
from pytorch_lightning import LightningModule, Trainer
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback

from src.models.diffusion import DiffusionModule, ConditionDiffusionModule
from src.models.vae import VAEModule


class GenSample(Callback):

    def __init__(self,
                 grid_shape: Tuple[int, int],
                 mean: float,
                 std: float,
                 log_cond: bool = False):
        self.grid_shape = grid_shape
        self.mean = mean
        self.std = std
        self.log_cond = log_cond
        self.images = {
            'train': None,
            'val': None,
            'test': None,
        }

        self.conds = {
            'train': None,
            'val': None,
            'test': None,
        }

    # train
    def on_train_epoch_start(self, trainer: Trainer,
                             pl_module: LightningModule) -> None:
        self.images['train'] = None
        self.conds['train'] = None

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        self.store_data(batch, mode='train')

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.sample(trainer, pl_module, mode='train')

    # validation
    def on_validation_epoch_start(self, trainer: Trainer,
                                  pl_module: LightningModule) -> None:
        self.images['val'] = None
        self.conds['val'] = None

    def on_validation_batch_end(self, trainer: Trainer,
                                pl_module: LightningModule, outputs: Any,
                                batch: Any, batch_idx: int,
                                dataloader_idx: int) -> None:
        self.store_data(batch, mode='val')

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule):
        self.sample(trainer, pl_module, mode='val')

    # test
    def on_test_epoch_start(self, trainer: Trainer,
                            pl_module: LightningModule) -> None:
        self.images['test'] = None
        self.conds['test'] = None

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs: Any, batch: Any, batch_idx: int,
                          dataloader_idx: int) -> None:
        self.store_data(batch, mode='test')

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.sample(trainer, pl_module, mode='test')

    def store_data(self, batch: Any, mode: str):
        if self.images[mode] is None:
            self.images[mode] = batch[0]
            self.conds[mode] = batch[1]
        elif self.images[mode].shape[
                0] < self.grid_shape[0] * self.grid_shape[1]:
            self.images[mode] = torch.cat((self.images[mode], batch[0]), dim=0)

            for key in self.conds[mode].keys():
                self.conds[mode][key] = torch.cat(
                    (self.conds[mode][key], batch[1][key]), dim=0)

    @torch.no_grad()  # for VAE forward
    def sample(self, trainer: Trainer, pl_module: LightningModule, mode: str):
        n_samples = self.grid_shape[0] * self.grid_shape[1]

        if isinstance(pl_module, VAEModule):
            targets = self.images[mode][:n_samples]

            with pl_module.ema_scope():
                preds = pl_module.net(targets)[0]  # remove grad
                samples = pl_module.net.sample(n_samples=n_samples,
                                               device=pl_module.device)

            targets = (targets * self.std + self.mean).clamp(0, 1)
            preds = (preds * self.std + self.mean).clamp(0, 1)

            self.log_sample([samples, preds, targets],
                            trainer=trainer,
                            nrow=self.grid_shape[0],
                            mode=mode,
                            caption=['samples', 'recons_img', 'target'])

            self.interpolation(pl_module, trainer, mode=mode)

        elif isinstance(pl_module, DiffusionModule):
            conds = None
            if isinstance(pl_module, ConditionDiffusionModule):
                from IPython import embed
                embed()
                for key in self.conds[mode].keys():
                    self.conds[mode][key] = self.conds[mode][key][:n_samples]
                conds = self.conds[mode]

            with pl_module.ema_scope():
                samples = pl_module.net.sample(num_sample=n_samples,
                                               device=pl_module.device,
                                               cond=conds)

            fakes = samples[-1]
            fakes = (fakes * self.std + self.mean).clamp(0, 1)

            reals = self.images[mode][:n_samples]
            reals = (reals * self.std + self.mean).clamp(0, 1)

            self.log_sample(
                [fakes, reals, conds] if self.log_cond else [fakes, reals],
                trainer=trainer,
                nrow=self.grid_shape[0],
                mode=mode,
                caption=['fake', 'real', 'cond']
                if self.log_cond else ['fake', 'real'])

            self.interpolation(pl_module, trainer, mode=mode)

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

    def interpolation(self, pl_module: LightningModule, trainer: Trainer,
                      mode: str):

        images = self.images[mode][0:2]

        if isinstance(pl_module, VAEModule):
            z, _ = pl_module.net.encode(images)
            z0, z1 = z[0], z[1]

            interpolated_z = []
            max_alpha = 24
            for alpha in range(1, max_alpha, 1):
                alpha /= max_alpha
                interpolated_z.append(z0 * (1 - alpha) + z1 * alpha)
            interpolated_z = torch.stack(interpolated_z, dim=0)

            interpolated_img = pl_module.net.decode(interpolated_z)

        elif isinstance(pl_module, DiffusionModule):
            from src.models.diffusion.sampler import DDPMSampler
            if isinstance(pl_module,
                          ConditionDiffusionModule) or not isinstance(
                              pl_module.net.sampler, DDPMSampler):
                return

            time_step = 50
            sample_steps = torch.tensor([time_step] * 2,
                                        dtype=torch.int64,
                                        device=pl_module.device)
            xt = pl_module.net.sampler.step(images, t=sample_steps)
            xt0, xt1 = xt[0], xt[1]

            interpolated_z = []
            max_alpha = 24
            for alpha in range(1, max_alpha, 1):
                alpha /= max_alpha
                interpolated_z.append(xt0 * (1 - alpha) + xt1 * alpha)
            interpolated_z = torch.stack(interpolated_z, dim=0)

            sample_steps = torch.arange(0, time_step, 1)
            gen_samples = pl_module.net.sampler.reverse_step(
                interpolated_z, sample_steps=sample_steps)
            interpolated_img = gen_samples[-1]

        interpolated_img = torch.cat(
            [images[0].unsqueeze(0), interpolated_img, images[1].unsqueeze(0)],
            dim=0)
        interpolated_img = (interpolated_img * self.std + self.mean).clamp(
            0, 1)
        interpolated_img = make_grid(interpolated_img, nrow=5)

        # logging
        trainer.logger.log_image(key=mode + '/interpolation',
                                 images=[interpolated_img])
