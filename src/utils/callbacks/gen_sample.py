from typing import Any, Tuple, List

import torch
from torch import Tensor
from pytorch_lightning import LightningModule, Trainer
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback

from src.models.diffusion import DiffusionModule, ConditionDiffusionModule
from src.models.vae import VAEModule


class GenSample(Callback):

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        mean: float,
        std: float,
        n_ensemble: int = 1,
    ):
        self.grid_shape = grid_shape
        self.mean = mean
        self.std = std
        self.n_ensemble = n_ensemble
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
        self.sample(pl_module, mode='train')

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
        self.sample(pl_module, mode='val')

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
        self.sample(pl_module, mode='test')

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
    def sample(self, pl_module: LightningModule, mode: str):
        # for eval before resuming training: store data not enough to grid
        if self.images[mode] == None:
            return

        n_samples = min(self.grid_shape[0] * self.grid_shape[1],
                        self.images[mode].shape[0])

        if isinstance(pl_module, VAEModule):
            targets = self.images[mode][:n_samples]

            with pl_module.ema_scope():
                preds = pl_module.net(targets)[0]  # remove grad
                samples = pl_module.net.sample(n_samples=n_samples,
                                               device=pl_module.device)

            targets = (targets * self.std + self.mean).clamp(0, 1)
            preds = (preds * self.std + self.mean).clamp(0, 1)

            self.log_sample([samples, preds, targets],
                            pl_module=pl_module,
                            nrow=self.grid_shape[0],
                            mode=mode,
                            caption=['samples', 'recons_img', 'target'])

            self.interpolation(pl_module=pl_module, mode=mode)

        elif isinstance(pl_module, DiffusionModule):
            # checking before resume training
            if self.images[mode] is None:
                return

            conds = None

            if isinstance(pl_module, ConditionDiffusionModule):
                conds = self.conds[mode].copy()
                for key in self.conds[mode].keys():
                    conds[key] = conds[key][:n_samples]

            reals = self.images[mode][:n_samples]

            with pl_module.ema_scope():
                fakes = []
                b, c, w, h = reals.shape
                for i in range(self.n_ensemble):
                    samples = pl_module.net.sample(num_sample=n_samples,
                                                   device=pl_module.device,
                                                   cond=conds.copy())
                    fakes.append(samples[-1])

            fakes = torch.cat(fakes, dim=0)
            fakes = fakes.reshape(self.n_ensemble, b, c, w, h).moveaxis(0, 1)

            # check variance
            if isinstance(
                    pl_module,
                    ConditionDiffusionModule) and 'image' in conds.keys():
                self.compute_variance(pl_module, reals.clone(), fakes.clone(),
                                      mode)

            fakes = fakes.mean(dim=1)
            fakes = (fakes * self.std + self.mean).clamp(0, 1)
            reals = (reals * self.std + self.mean).clamp(0, 1)

            self.log_sample(
                [fakes, reals, conds['image']] if conds is not None
                and 'image' in conds.keys() else [fakes, reals],
                pl_module=pl_module,
                nrow=self.grid_shape[0],
                mode=mode,
                caption=['fake', 'real', 'cond'] if conds is not None
                and 'image' in conds.keys() else ['fake', 'real'])

            self.interpolation(pl_module, mode=mode)

    def compute_variance(self, pl_module: ConditionDiffusionModule,
                         reals: Tensor, fakes: Tensor, mode: str):
        _, c, w, h = reals.shape
        ensemble = fakes.mean(dim=1)
        variance = fakes.var(dim=1)

        # log average variance on the whole image
        pl_module.log(mode + '/avg_variance',
                      variance.mean(),
                      on_step=False,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)

        # only log 6 images
        n_images = 6
        if variance.shape[0] < n_images: return

        # log heatmap
        self.log_heatmap(variance, pl_module, mode, n_images)

        ensemble = ensemble[:n_images]
        variance = variance[:n_images]
        fakes = fakes[:n_images]
        reals = reals[:n_images]

        # log ensemble
        fakes = fakes.reshape(-1, c, w, h)
        images = [
            fakes, ensemble, variance, reals,
            self.conds[mode]['image'][:n_images]
        ]
        images = [
            make_grid(image, nrow=int(image.shape[0] / n_images))
            for image in images
        ]
        captions = ['fake', 'ensemble', 'variance', 'real', 'cond']

        if images[-1].shape[0] == 4:
            t1 = images[-1][0:1, ...]
            t1ce = images[-1][1:2, ...]
            t2 = images[-1][2:3, ...]
            flair = images[-1][3:, ...]

            images[-1] = t1 / t1.max()
            images += [t1ce / t1ce.max(), t2 / t2.max(), flair / flair.max()]

            caption[-1] = ['t1']
            caption += ['t1ce', 't2', 'flair']

        pl_module.logger.log_image(key=mode + '/variance',
                                   images=images,
                                   caption=captions)

    def log_heatmap(self,
                    variance: Tensor,
                    pl_module: LightningModule,
                    mode: str,
                    n_images=6):

        import matplotlib.pyplot as plt
        import seaborn as sns

        data = variance[:n_images].mean(dim=1)
        plt.figure(figsize=(30, 15))
        for i in range(n_images):
            plt.subplot(2, n_images // 2, i + 1)
            sns.heatmap(data=data[i].cpu())
            plt.axis('off')
        pl_module.logger.log_image(key=mode + '/heatmap', images=[plt])

    def log_sample(self,
                   images: Tensor,
                   pl_module: LightningModule,
                   nrow: int,
                   mode: str,
                   caption=List[str]):
        if 'cond' in caption and images[-1].shape[1] == 4:
            t1 = images[-1][:, 0:1, :, :]
            t1ce = images[-1][:, 1:2, :, :]
            t2 = images[-1][:, 2:3, :, :]
            flair = images[-1][:, 3:, :, :]

            images[-1] = t1 / t1.max()
            images += [t1ce / t1ce.max(), t2 / t2.max(), flair / flair.max()]

            caption[-1] = ['t1']
            caption += ['t1ce', 't2', 'flair']

        images = [make_grid(image, nrow=nrow) for image in images]

        # logging
        pl_module.logger.log_image(key=mode + '/inference',
                                   images=images,
                                   caption=caption)

    def interpolation(self, pl_module: LightningModule, mode: str):

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
        pl_module.logger.log_image(key=mode + '/interpolation',
                                   images=[interpolated_img])
