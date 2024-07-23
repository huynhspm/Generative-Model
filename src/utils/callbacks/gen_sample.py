from typing import Any, Tuple, List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import Tensor
from pytorch_lightning import LightningModule, Trainer
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback

from src.models.gan import GANModule, ConditionGANModule
from src.models.diffusion import DiffusionModule, ConditionDiffusionModule
from src.models.vae import VAEModule
from src.models.unet import UNetModule


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

    # train
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        if batch_idx == 0:
            self.sample(pl_module, batch, mode="train")

    # validation
    def on_validation_batch_end(self, trainer: Trainer,
                                pl_module: LightningModule, outputs: Any,
                                batch: Any, batch_idx: int,
                                dataloader_idx: int) -> None:
        if batch_idx == 0:
            self.sample(pl_module, batch, mode="val")

    # test
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs: Any, batch: Any, batch_idx: int,
                          dataloader_idx: int) -> None:
        if batch_idx == 0:
            self.sample(pl_module, batch, mode="test")

    def rescale(self, image: Tensor):
        #convert range (-1, 1) to (0, 1)
        return (image * self.std + self.mean).clamp(0, 1)

    @torch.no_grad()  # for VAE forward
    def sample(self, pl_module: LightningModule, batch: Any, mode: str):

        # avoid out of memory
        n_samples = min(self.grid_shape[0] * self.grid_shape[1],
                        batch[0].shape[0])

        with pl_module.ema_scope():
            if isinstance(pl_module, VAEModule):
                targets = batch[0][:n_samples]
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
                reals = batch[0][:n_samples]

                conds = None
                if isinstance(pl_module, ConditionDiffusionModule):
                    conds = {key: value[:n_samples] for key, value in batch[1].items()}

                fakes = []
                for _ in range(self.n_ensemble):
                    samples = pl_module.net.sample(
                        num_sample=n_samples,
                        device=pl_module.device,
                        cond=conds.copy() if isinstance(
                            pl_module, ConditionDiffusionModule) else None)
                    fakes.append(samples[-1])  # b, c, w, h

                fakes = torch.stack(fakes, dim=1)  # b, n ,c, w, h

                fakes = self.rescale(fakes)
                reals = self.rescale(reals)

                # check variance for segmentation task
                if self.n_ensemble > 1:
                    self.compute_variance(pl_module, reals, fakes, conds, mode)

                # ensemble. If generation task, only 1 sample -> ensemble to unsqueeze dim 1
                fakes = fakes.mean(dim=1)  # b, c, w, h

                self.log_sample(
                    [fakes, reals, conds['image']] if conds is not None
                    and 'image' in conds.keys() else [fakes, reals],
                    pl_module=pl_module,
                    nrow=self.grid_shape[0],
                    mode=mode,
                    caption=['fake', 'real', 'cond'] if conds is not None
                    and 'image' in conds.keys() else ['fake', 'real'])
            
            elif isinstance(pl_module, GANModule):
                reals = batch[0][:n_samples]

                conds = None
                if isinstance(pl_module, ConditionGANModule):
                    conds = {key: value[:n_samples] for key, value in batch[1].items()}

                fakes = pl_module.net.sample(num_sample=n_samples, 
                                             device=pl_module.device, 
                                             cond=conds if isinstance(
                                                 pl_module, ConditionGANModule) else None)

                reals = self.rescale(reals)
                fakes = self.rescale(fakes)

                self.log_sample([fakes, reals],
                                pl_module=pl_module,
                                nrow=self.grid_shape[0],
                                mode=mode,
                                caption=['fake', 'real'])

            elif isinstance(pl_module, UNetModule):
                masks = batch[0][:n_samples]
                images = batch[1]["image"][:n_samples]
                preds = pl_module.predict(images)

                self.log_sample([preds, masks, images],
                                pl_module=pl_module,
                                nrow=self.grid_shape[0],
                                mode=mode,
                                caption=['preds', 'mask', "image"])


            else:
                raise NotImplementedError('This module is not implemented')
            
        if batch[0].shape[0] > 1:
            self.interpolation(pl_module=pl_module, images=batch[0][:2],mode=mode)

    def compute_variance(self,
                         pl_module: ConditionDiffusionModule,
                         reals: Tensor,
                         fakes: Tensor,
                         conds: Dict[str, Tensor],
                         mode: str, 
                         n_images: int = 6):
        ensemble = fakes.mean(dim=1)
        fake_variance = ((fakes > 0.5).to(torch.float32)).var(dim=1)
        _, c, w, h = reals.shape

        # only get n_images to log variance
        n_images = min(n_images, reals.shape[0])
        _reals = reals[:n_images]
        ensemble = ensemble[:n_images]
        fake_variance = fake_variance[:n_images]
        _conds = conds['image'][:n_images]
        _fakes = fakes[:n_images].reshape(-1, c, w, h)

        # log heatmap
        self.log_heatmap(fake_variance, pl_module, mode, 'fake')

        
        # log ensemble
        images = [_fakes, ensemble, fake_variance, _reals, _conds]
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

        if 'masks' in conds.keys():
            # batch, 4, w, h
            masks = conds['masks'][:n_images]
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

    def interpolation(self, pl_module: LightningModule, images: Tensor, mode: str):
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

                # todo fix interpolation for diffusion
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
                gen_samples = pl_module.net.sample(
                    interpolated_z, sample_steps=sample_steps)
                interpolated_img = gen_samples[-1]
            else:
                return 
            

        interpolated_img = torch.cat(
            [images[0].unsqueeze(0), interpolated_img, images[1].unsqueeze(0)],
            dim=0)
        interpolated_img = self.rescale(interpolated_img)
        interpolated_img = make_grid(interpolated_img, nrow=5, pad_value=1)

        # logging
        pl_module.logger.log_image(key=mode + '/interpolation',
                                   images=[interpolated_img])
