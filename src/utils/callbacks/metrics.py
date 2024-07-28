from typing import Any, Dict
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio, FrechetInceptionDistance,
                                InceptionScore)

from torchmetrics import Dice, JaccardIndex, MeanMetric

from src.models.gan import GANModule, ConditionGANModule
from src.models.diffusion import DiffusionModule, ConditionDiffusionModule
from src.models.vae import VAEModule
from src.models.unet import UNetModule


class Metrics(Callback):

    def __init__(
        self,
        ssim: StructuralSimilarityIndexMeasure | None = None,
        psnr: PeakSignalNoiseRatio | None = None,
        fid: FrechetInceptionDistance | None = None,
        IS: InceptionScore | None = None,
        dice: Dice | None = None,
        iou: JaccardIndex | None = None,
        image_variance: MeanMetric | None = None,
        boundary_variance: MeanMetric | None = None,
        mean: float = 0.5,
        std: float = 0.5,
        n_ensemble: int | None = None,
    ) -> None:
        """_summary_

        Args:
            ssim (StructuralSimilarityIndexMeasure | None, optional): metrics for VAE. Defaults to None.
            psnr (PeakSignalNoiseRatio | None, optional): metrics for VAE. Defaults to None.
            fid (FrechetInceptionDistance | None, optional): metrics for generation task (diffusion, gan). Defaults to None.
            IS (InceptionScore | None, optional): metrics for generation task (not good - weight of inception-v3). Defaults to None.
            dice (Dice | None, optional): metrics for segmentation task. Defaults to None.
            iou (JaccardIndex | None, optional): metrics for segmentation task. Defaults to None.
            image_variance (MeanMetric | None, optional): _description_. Defaults to None.
            boundary_variance (MeanMetric | None, optional): _description_. Defaults to None.
            mean (float, optional): to convert image into (0, 1). Defaults to 0.5.
            std (float, optional): to convert image into (0, 1). Defaults to 0.5.
            n_ensemble (int, optional): for segmentation with diffusion model. Defaults to 1.
        """

        self.ssim = ssim
        self.psnr = psnr
        self.fid = fid
        self.IS = IS
        self.dice = dice
        self.iou = iou
        self.image_variance = image_variance
        self.boundary_variance = boundary_variance

        self.mean = mean
        self.std = std
        self.n_ensemble = n_ensemble

    # def on_train_epoch_start(self, trainer: Trainer,
    #                          pl_module: LightningModule) -> None:
    #     self.reset_metrics()

    # def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
    #                        outputs: STEP_OUTPUT, batch: Any,
    #                        batch_idx: int) -> None:
    #     self.update_metrics(pl_module, batch)

    # def on_train_epoch_end(self, trainer: Trainer,
    #                        pl_module: LightningModule) -> None:
    #     self.log_metrics(pl_module, mode='train')

    def on_validation_epoch_start(self, trainer: Trainer,
                                  pl_module: LightningModule) -> None:
        self.reset_metrics()

    def on_validation_batch_end(self, trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: STEP_OUTPUT | None, batch: Any,
                                batch_idx: int, dataloader_idx: int) -> None:
        self.update_metrics(pl_module, batch)

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule) -> None:
        self.log_metrics(pl_module, mode='val')

    def on_test_epoch_start(self, trainer: Trainer,
                            pl_module: LightningModule) -> None:
        self.reset_metrics()

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs: STEP_OUTPUT | None, batch: Any,
                          batch_idx: int, dataloader_idx: int) -> None:
        self.update_metrics(pl_module, batch)

    def on_test_epoch_end(self, trainer: Trainer,
                          pl_module: LightningModule) -> None:
        self.log_metrics(pl_module, mode='test')

    def rescale(self, image: Tensor) -> Tensor:
        #convert range (-1, 1) to (0, 1)
        return (image * self.std + self.mean).clamp(0, 1)

    def reset_metrics(self) -> None:
        if self.ssim is not None:
            self.ssim.reset()

        if self.psnr is not None:
            self.psnr.reset()

        if self.fid is not None:
            self.fid.reset()

        if self.IS is not None:
            self.IS.reset()

        if self.dice is not None:
            self.dice.reset()

        if self.iou is not None:
            self.iou.reset()

        if self.image_variance is not None:
            self.image_variance.reset()

        if self.boundary_variance is not None:
            self.boundary_variance.reset()
            
    def infer(self, pl_module: LightningModule, batch: Any) -> Tensor:
        
        if isinstance(pl_module, UNetModule):
            preds = pl_module.predict(batch[1]["image"])
            return preds # range [0, 1]

        elif isinstance(pl_module, VAEModule):
            preds = pl_module.predict(batch[0])
            return preds # range [0, 1]

        elif isinstance(pl_module, DiffusionModule):
            fakes = []
            for _ in range(self.n_ensemble):
                cond=batch[1].copy() if isinstance(pl_module, ConditionDiffusionModule) else None
                samples = pl_module.predict(num_sample=batch[0].shape[0],
                                            device=pl_module.device,
                                            cond=cond) #  range (-1, 1)
                fakes.append(samples[-1])  # [b, c, w, h]
            
            fakes = torch.stack(fakes, dim=1)  # (b, n, c, w, h)
            
            return self.rescale(fakes) # range [0, 1]

        elif isinstance(pl_module, GANModule):
            cond=batch[1] if isinstance(pl_module, ConditionGANModule) else None
            samples = pl_module.predict(num_sample=batch[0].shape[0],
                                        device=pl_module.device,
                                        cond=cond) # range [-1, 1]
            return self.rescale(samples) # range [0, 1]

        else:
            raise NotImplementedError('This module is not implemented')

    def update_metrics(self, pl_module: LightningModule, batch: Any) -> None:
        
        targets = self.rescale(batch[0]) # range [0, 1]
        preds = self.infer(pl_module, batch)

        if self.image_variance is not None or self.boundary_variance is not None:
            self.update_variance(preds, pl_module.device)

        if len(preds.shape) == 5: # [b, n, c, w, h]
            preds = preds.mean(dim=1)

        # update
        if self.ssim is not None:
            self.ssim.to(pl_module.device)
            self.ssim.update(preds, targets)
            self.ssim.to('cpu')

        if self.psnr is not None:
            self.psnr.to(pl_module.device)
            self.psnr.update(preds, targets)
            self.psnr.to('cpu')

        if self.dice is not None or self.iou is not None:
            preds = preds.to(torch.int64)
            targets = targets.to(torch.int64)

            if self.dice is not None:
                self.dice.to(pl_module.device)
                self.dice.update(preds, targets)
                self.dice.to('cpu')

            if self.iou is not None:
                self.iou.to(pl_module.device)
                self.iou.update(preds, targets)
                self.iou.to('cpu')

        if self.fid is not None or self.IS is not None:
            if preds.shape[1] == 1:
                # gray to rgb image
                fakes = torch.cat([preds, preds, preds], dim=1)
                reals = torch.cat([targets, targets, targets], dim=1)
            
            reals = torch.nn.functional.interpolate(reals,
                                                    size=(299, 299),
                                                    mode='bilinear')
            fakes = torch.nn.functional.interpolate(fakes,
                                                    size=(299, 299),
                                                    mode='bilinear')

            if self.fid is not None:
                self.fid.to(pl_module.device)
                self.fid.update(reals, real=True)
                self.fid.update(fakes, real=False)
                self.fid.to('cpu')

            if self.IS is not None:
                self.IS.to(pl_module.device)
                self.IS.update(fakes)
                self.IS.to('cpu')

    def update_variance(self, preds: Tensor, device: torch.device) -> None:

        # (b, n, c, w, h) -> (b, c, w, h)
        variance = ((preds > 0.5).to(torch.float32)).var(dim=1)

        if self.image_variance is not None:
            self.image_variance.to(device)
            self.image_variance.update(variance.mean())
            self.image_variance.to('cpu')

        if self.boundary_variance is not None:
            boundary = variance > 0
            boundary_variance = ((variance).sum(dim=[1, 2, 3]) +
                                 1) / (boundary.sum(dim=[1, 2, 3]) + 1)
            self.boundary_variance.to(device)
            self.boundary_variance.update(boundary_variance)
            self.boundary_variance.to('cpu')

    def log_metrics(self, pl_module: LightningModule, mode: str) -> None:

        if self.image_variance is not None:
            self.image_variance.to(pl_module.device)
            pl_module.log(mode + '/image_variance',
                          self.image_variance.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.image_variance.to('cpu')

        if self.boundary_variance is not None:
            self.boundary_variance.to(pl_module.device)
            pl_module.log(mode + '/boundary_variance',
                          self.boundary_variance.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.boundary_variance.to('cpu')

        if self.ssim is not None:
            self.ssim.to(pl_module.device)
            pl_module.log(mode + '/ssim',
                          self.ssim.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.ssim.to('cpu')

        if self.psnr is not None:
            self.psnr.to(pl_module.device)
            pl_module.log(mode + '/psnr',
                          self.psnr.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.psnr.to('cpu')

        if self.dice is not None:
            self.dice.to(pl_module.device)
            pl_module.log(mode + '/dice',
                          self.dice.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.dice.to('cpu')

        if self.iou is not None:
            self.iou.to(pl_module.device)
            pl_module.log(mode + '/iou',
                          self.iou.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.iou.to('cpu')

        if self.fid is not None:
            self.fid.to(pl_module.device)
            pl_module.log(mode + '/fid',
                          self.fid.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.fid.to('cpu')

        if self.IS is not None:
            self.IS.to(pl_module.device)
            mean, std = self.IS.compute()
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
            self.IS.to('cpu')
