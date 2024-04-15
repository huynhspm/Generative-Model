from typing import Any, Tuple, List

import matplotlib.pyplot as plt
import seaborn as sns
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
        """_summary_

        Args:
            grid_shape (Tuple[int, int]): _description_
            mean (float): _description_
            std (float): _description_
            n_ensemble (int, optional): _description_. Defaults to 1.

        Raises:
            NotImplementedError: _description_
        """
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

    def rescale(self, image: Tensor):
        #convert range (-1, 1) to (0, 1)
        return (image * self.std + self.mean).clamp(0, 1)

    @torch.no_grad()  # for VAE forward
    def sample(self, pl_module: LightningModule, mode: str):
        # for eval before resuming training: store data not enough to grid
        if self.images[mode] is None:
            return

        n_samples = min(self.grid_shape[0] * self.grid_shape[1],
                        self.images[mode].shape[0])

        with pl_module.ema_scope():
            if isinstance(pl_module, VAEModule):
                targets = self.images[mode][:n_samples]
                preds, _ = pl_module.net(targets)  # remove grad
                samples = pl_module.net.sample(n_samples=n_samples,
                                               device=pl_module.device)

                targets = self.rescale(targets)
                preds = self.rescale(preds)

                self.log_sample([samples, preds, targets],
                                pl_module=pl_module,
                                nrow=self.grid_shape[0],
                                mode=mode,
                                caption=['samples', 'recons_img', 'target'])

            elif isinstance(pl_module, DiffusionModule):
                reals = self.images[mode][:n_samples]
                conds = {
                    key: self.conds[mode][key][:n_samples]
                    for key in self.conds[mode].keys()
                } if isinstance(pl_module, ConditionDiffusionModule) else None

                fakes = []
                for _ in range(self.n_ensemble):
                    samples = pl_module.net.sample(
                        num_sample=n_samples,
                        device=pl_module.device,
                        cond=None if conds is None else conds.copy())
                    fakes.append(samples[-1])  # b, c, w, h

                fakes = torch.stack(fakes, dim=1)  # b, n ,c, w, h

                fakes = self.rescale(fakes)
                reals = self.rescale(reals)

                # check variance
                if self.n_ensemble > 1:
                    self.compute_variance(pl_module, reals.clone(),
                                          fakes.clone(), mode)

                # ensemble
                fakes = fakes.mean(dim=1)  # b, c, w, h

                self.log_sample(
                    [fakes, reals, conds['image']] if conds is not None
                    and 'image' in conds.keys() else [fakes, reals],
                    pl_module=pl_module,
                    nrow=self.grid_shape[0],
                    mode=mode,
                    caption=['fake', 'real', 'cond'] if conds is not None
                    and 'image' in conds.keys() else ['fake', 'real'])

        self.interpolation(pl_module=pl_module, mode=mode)

    def compute_variance(self,
                         pl_module: ConditionDiffusionModule,
                         reals: Tensor,
                         fakes: Tensor,
                         mode: str,
                         n_images: int = 6):
        ensemble = fakes.mean(dim=1)
        fake_variance = fakes.var(dim=1)

        if fake_variance.shape[0] < n_images: return

        ensemble = ensemble[:n_images]
        fake_variance = fake_variance[:n_images]
        fakes = fakes[:n_images]
        reals = reals[:n_images]
        conds = self.conds[mode]['image'][:n_images]
        conds = self.rescale(conds)

        # log heatmap
        self.log_heatmap(fake_variance, pl_module, mode, 'fake')

        # log ensemble
        _, c, w, h = reals.shape
        fakes = fakes.reshape(-1, c, w, h)
        images = [fakes, ensemble, fake_variance, reals, conds]
        captions = ['fake', 'ensemble', 'fake-variance', 'real', 'cond']

        if images[-1].shape[0] == 4:
            t1 = images[-1][0:1, ...]
            t1ce = images[-1][1:2, ...]
            t2 = images[-1][2:3, ...]
            flair = images[-1][3:4, ...]

            images[-1] = t1
            images += [t1ce, t2, flair]

            caption[-1] = ['t1']
            caption += ['t1ce', 't2', 'flair']

        if 'masks' in self.conds[mode].keys():
            # batch, 4, w, h
            masks = self.conds[mode]['masks'][:n_images]
            masks = self.rescale(masks)

            # batch, 4, c, w, h -> b*4, c ,w, h
            masks = masks.unsqueeze(dim=2)
            real_variance = masks.var(dim=1)
            masks = masks.reshape(-1, c, w, h)

            images += [masks, real_variance]
            caption += ['real-masks', 'real-variance']
            self.log_heatmap(real_variance, pl_module, mode, 'real')

        images = [
            make_grid(image, nrow=int(image.shape[0] / n_images), pad_value=1)
            for image in images
        ]

        pl_module.logger.log_image(key=mode + '/variance',
                                   images=images,
                                   caption=captions)

    def log_heatmap(
        self,
        variance: Tensor,
        pl_module: LightningModule,
        mode: str,
        caption: str,
    ):
        data = variance.mean(dim=1)  # b, c, w, h -> b, w, h
        plt.figure(figsize=(30, 15))
        for i in range(variance.shape[0]):
            plt.subplot(2, variance.shape[0] // 2, i + 1)
            sns.heatmap(data=data[i].cpu())
            plt.axis('off')
        pl_module.logger.log_image(key=mode + f'/{caption}_heatmap',
                                   images=[plt])

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

            images[-1] = t1
            images += [t1ce, t2, flair]

            caption[-1] = ['t1']
            caption += ['t1ce', 't2', 'flair']

        images = [make_grid(image, nrow=nrow, pad_value=1) for image in images]

        # logging
        pl_module.logger.log_image(key=mode + '/inference',
                                   images=images,
                                   caption=caption)

    def interpolation(self, pl_module: LightningModule, mode: str):

        images = self.images[mode][0:2]
        max_step = 24
        alphas = [t / max_step for t in range(1, max_step, 1)]

        # auto use ema
        with pl_module.ema_scope():

            if isinstance(pl_module, VAEModule):
                z, _ = pl_module.net.encode(images)
                z0, z1 = z[0], z[1]

                interpolated_z = []
                for alpha in alphas:
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
                for alpha in alphas:
                    interpolated_z.append(xt0 * (1 - alpha) + xt1 * alpha)
                interpolated_z = torch.stack(interpolated_z, dim=0)

                sample_steps = torch.arange(0, time_step, 1)
                gen_samples = pl_module.net.sampler.reverse_step(
                    interpolated_z, sample_steps=sample_steps)
                interpolated_img = gen_samples[-1]

        interpolated_img = torch.cat(
            [images[0].unsqueeze(0), interpolated_img, images[1].unsqueeze(0)],
            dim=0)
        interpolated_img = self.rescale(interpolated_img)
        interpolated_img = make_grid(interpolated_img, nrow=5, pad_value=1)

        # logging
        pl_module.logger.log_image(key=mode + '/interpolation',
                                   images=[interpolated_img])
