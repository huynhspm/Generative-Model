from typing import List, Any, Tuple, Optional

import torch
from torch import Tensor
from pytorch_lightning import LightningModule, Trainer 
from torchmetrics.image import (FrechetInceptionDistance, 
                                InceptionScore, 
                                StructuralSimilarityIndexMeasure, 
                                PeakSignalNoiseRatio)
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback

from src.models.diffusion import DiffusionModule, ConditionDiffusionModule
from src.models.vae import VAEModule

class GenSample(Callback):

    def __init__(self, 
                 grid_shape: Tuple[int, int], 
                 gen_type: str,
                 mean: List[float], 
                 std: List[float], 
                 metrics: List[str],
                 feature: Optional[int] = None,
                 range: float = 2.0):
        self.grid_shape = grid_shape
        self.gen_type = gen_type
        self.mean = Tensor(mean).reshape(1, -1, 1, 1)
        self.std = Tensor(std).reshape(1, -1, 1, 1)
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
        self.metrics = metrics
        
        if self.metrics is not None and 'fid' in self.metrics:
            self.fid_metric = FrechetInceptionDistance(feature=feature,
                                                       normalize=True)

        if self.metrics is not None and 'is' in self.metrics:
            self.is_metric = InceptionScore(feature=feature,
                                            normalize=True)

        if self.metrics is not None and 'ssim' in self.metrics:
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=range)
        
        if self.metrics is not None and 'psnr' in self.metrics:
            self.psnr_metric = PeakSignalNoiseRatio(data_range=range)
    
    @torch.no_grad()
    def reset_metric(self) -> None:
        if self.metrics is not None and 'fid' in self.metrics: 
            self.fid_metric.reset()
        
        if self.metrics is not None and 'is' in self.metrics:
            self.is_metric.reset()

        if self.metrics is not None and 'ssim' in self.metrics:
            self.ssim_metric.reset()

        if self.metrics is not None and 'psnr' in self.metrics:
            self.psnr_metric.reset()
            

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.reset_metric()
    
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.reset_metric()
    
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.reset_metric()
    
    def on_train_epoch_end(self, 
                           trainer: Trainer, 
                           pl_module: LightningModule):
        self.sample(trainer, pl_module, mode='train')

    def on_validation_epoch_end(self, 
                                trainer: Trainer,
                                pl_module: LightningModule):
        self.sample(trainer, pl_module, mode='val')

    def on_test_epoch_end(self, 
                          trainer: Trainer,
                          pl_module: LightningModule):
        self.sample(trainer, pl_module, mode='test')

    @torch.no_grad()
    def sample(self, 
               trainer: Trainer, 
               pl_module: LightningModule,
               mode: str):
        n_samples = self.grid_shape[0] * self.grid_shape[1]
        self.mean = self.mean.to(pl_module.device)
        self.std = self.std.to(pl_module.device)

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

            if self.metrics is not None and 'ssim' in self.metrics:
                self.ssim_metric.to(pl_module.device)
                self.log_ssim_metric(preds, targets, pl_module, mode=mode)

            if self.metrics is not None and 'psnr' in self.metrics:
                self.psnr_metric.to(pl_module.device)
                self.log_psnr_metric(preds, targets, pl_module, mode=mode)

            self.interpolation(self.images[mode][0:2],
                               pl_module, 
                               trainer,
                               mode=mode)

        elif isinstance(pl_module, DiffusionModule):
            cond = None
            if isinstance(pl_module, ConditionDiffusionModule):
                cond = self.labels[mode][:n_samples]
            
            if pl_module.use_ema:
                with pl_module.ema_scope():
                    samples = pl_module.net.get_p_sample(
                        num_sample=n_samples,
                        device=pl_module.device,
                        cond=cond)
            else:
                samples = pl_module.net.get_p_sample(
                        num_sample=n_samples,
                        device=pl_module.device,
                        cond=cond)

            fakes = samples[-1]
            fakes = (fakes * self.std + self.mean).clamp(0, 1)

            reals = self.images[mode][:n_samples]
            reals = (reals * self.std + self.mean).clamp(0, 1)

            self.log_sample([fakes, reals], 
                            trainer=trainer, 
                            nrow=self.grid_shape[0], 
                            mode=mode, 
                            caption=['fake', 'real'])

            if self.metrics is not None and 'fid' in self.metrics:
                self.fid_metric.to(pl_module.device)
                self.log_fid_metric(fakes, reals, pl_module, mode=mode)

            if self.metrics is not None and 'is' in self.metrics:
                self.is_metric.to(pl_module.device)
                self.log_is_metric(fakes, pl_module, mode=mode)

            if self.metrics is not None and 'ssim' in self.metrics:
                self.ssim_metric.to(pl_module.device)
                self.log_ssim_metric(fakes, reals, pl_module, mode=mode)

            if self.metrics is not None and 'psnr' in self.metrics:
                self.psnr_metric.to(pl_module.device)
                self.log_psnr_metric(fakes, reals, pl_module, mode=mode)
                
        self.images[mode] = None
        self.labels[mode] = None

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

    def log_fid_metric(self, 
                       fake: Tensor, 
                       real: Tensor,
                       pl_module: LightningModule, 
                       mode: str):
        # reset
        self.fid_metric.reset()
        self.fid_metric.to(pl_module.device)

        # update
        self.fid_metric.update(fake, real=False)
        self.fid_metric.update(real, real=True)

        # logging
        pl_module.log(mode + '/fid',
                      self.fid_metric.compute(),
                      on_step=False,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)

    def log_is_metric(self, 
                      fake: Tensor, 
                      pl_module: LightningModule,
                      mode: str):

        # update
        self.is_metric.update(fake)

        #logging
        mean, std = self.is_metric.compute()
        range = {
            'min': mean - std,
            'max': mean + std,
        }

        pl_module.log(mode + '/is', 
                      range,
                      on_step=False,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)

    def log_ssim_metric(self, 
                       fakes: Tensor, 
                       reals: Tensor,
                       pl_module: LightningModule, 
                       mode: str):

        # update
        self.ssim_metric.update(fakes, reals)

        # logging
        pl_module.log(mode + '/ssim',
                      self.ssim_metric.compute(),
                      on_step=False,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)
    
    def log_psnr_metric(self, 
                       fakes: Tensor, 
                       reals: Tensor,
                       pl_module: LightningModule, 
                       mode: str):

        # update
        self.psnr_metric.update(fakes, reals)

        # logging
        pl_module.log(mode + '/psnr',
                      self.psnr_metric.compute(),
                      on_step=False,
                      on_epoch=True,
                      prog_bar=False,
                      sync_dist=True)
        
    def interpolation(self, 
                      images: Tensor,
                      pl_module: VAEModule, 
                      trainer: Trainer,
                      mode: str):
        self.mean = self.mean.to(pl_module.device)
        self.std = self.std.to(pl_module.device)

        z, _ = pl_module.net.encode(images)
        images = (images * self.std + self.mean).clamp(0, 1)
        z0, z1 = z[0].unsqueeze(0), z[1].unsqueeze(0)
        image0, image1 = images[0].unsqueeze(0), images[1].unsqueeze(0)
        
        interpolated_img = [image0]

        max_lambda = 34
        for alpha in range(0, max_lambda):
            alpha /= max_lambda
            interpolated_z = z0 * (1 - alpha) + z1 * alpha 
            out = pl_module.net.decode(interpolated_z)
            out = (out * self.std + self.mean).clamp(0, 1)
            interpolated_img.append(out)
        interpolated_img.append(image1)
        interpolated_img = torch.concatenate(interpolated_img, dim=0)
        interpolated_img = make_grid(interpolated_img, nrow=6)
        
        # logging
        trainer.logger.log_image(key=mode + '/interpolation',
                                 images=[interpolated_img])

    def on_train_batch_end(self, 
                           trainer: Trainer, 
                           pl_module: LightningModule,
                           outputs: Any, 
                           batch: Any, 
                           batch_idx: int) -> None:
       self.store_data(batch, mode='train')

    def on_validation_batch_end(self, 
                                trainer: Trainer,
                                pl_module: LightningModule, 
                                outputs: Any,
                                batch: Any, 
                                batch_idx: int,
                                dataloader_idx: int) -> None:
        self.store_data(batch, mode='val')

    def on_test_batch_end(self, 
                          trainer: Trainer,
                          pl_module: LightningModule, 
                          outputs: Any,
                          batch: Any, 
                          batch_idx: int,
                          dataloader_idx: int) -> None:
        self.store_data(batch, mode='test')

    def store_data(self, 
                   batch: Any, 
                   mode: str):
        if self.images[mode] == None:
            self.images[mode], self.labels[mode] = batch
        elif self.images[mode].shape[0] < self.grid_shape[0] * self.grid_shape[1]:
            self.images[mode] = torch.cat((self.images[mode], batch[0]), dim=0)
            self.labels[mode] = torch.cat((self.labels[mode], batch[1]), dim=0)