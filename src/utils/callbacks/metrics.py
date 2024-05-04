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

from src.models.diffusion import DiffusionModule, ConditionDiffusionModule
from src.models.vae import VAEModule


class Metrics(Callback):

    def __init__(
        self,
        ssim: StructuralSimilarityIndexMeasure | None = None,
        psnr: PeakSignalNoiseRatio | None = None,
        fid: FrechetInceptionDistance | None = None,
        IS: InceptionScore | None = None,
        dice: Dice | None = None,
        iou: JaccardIndex | None = None,
        mean_variance: MeanMetric | None = None,
        mean_boundary_variance: MeanMetric | None = None,
        boundary_threshold: float = 0.05,
        mean: float = 0.5,
        std: float = 0.5,
        n_ensemble: int = 1,
    ):
        """_summary_

        Args:
            ssim (StructuralSimilarityIndexMeasure | None, optional): metrics for VAE. Defaults to None.
            psnr (PeakSignalNoiseRatio | None, optional): metrics for VAE. Defaults to None.
            fid (FrechetInceptionDistance | None, optional): metrics for generation task (diffusion, gan). Defaults to None.
            IS (InceptionScore | None, optional): metrics for generation task (not good - weight of inception-v3). Defaults to None.
            dice (Dice | None, optional): metrics for segmentation task. Defaults to None.
            iou (JaccardIndex | None, optional): metrics for segmentation task. Defaults to None.
            mean_variance (MeanMetric | None, optional): _description_. Defaults to None.
            mean_boundary_variance (MeanMetric | None, optional): _description_. Defaults to None.
            boundary_threshold (float, optional): _description_. Defaults to 0.05.
            mean (float, optional): to convert image into (0, 1). Defaults to 0.5.
            std (float, optional): to convert image into (0, 1). Defaults to 0.5.
            n_ensemble (int, optional): _description_. Defaults to 1.
        """

        self.ssim = ssim
        self.psnr = psnr
        self.fid = fid
        self.IS = IS
        self.dice = dice
        self.iou = iou
        self.mean_variance = mean_variance

        self.mean_boundary_variance = mean_boundary_variance
        self.boundary_threshold = boundary_threshold

        self.mean = mean
        self.std = std
        self.n_ensemble = n_ensemble

    # def on_train_epoch_start(self, trainer: Trainer,
    #                          pl_module: LightningModule) -> None:
    #     self.reset_metrics()

    # def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
    #                        outputs: STEP_OUTPUT, batch: Any,
    #                        batch_idx: int) -> None:
    #     reals, conds = batch
    #     fakes = self.get_sample(pl_module, reals, conds)
    #     self.update_metrics(reals, fakes, device=pl_module.device)

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
        reals, conds = batch
        fakes = self.get_sample(pl_module, reals, conds)
        self.update_metrics(reals, fakes, device=pl_module.device)

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule) -> None:
        self.log_metrics(pl_module, mode='val')

    def on_test_epoch_start(self, trainer: Trainer,
                            pl_module: LightningModule) -> None:
        self.reset_metrics()

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs: STEP_OUTPUT | None, batch: Any,
                          batch_idx: int, dataloader_idx: int) -> None:
        reals, conds = batch
        fakes = self.get_sample(pl_module, reals, conds)
        self.update_metrics(reals, fakes, device=pl_module.device)

    def on_test_epoch_end(self, trainer: Trainer,
                          pl_module: LightningModule) -> None:
        self.log_metrics(pl_module, mode='test')

    def rescale(self, image: Tensor):
        #convert range (-1, 1) to (0, 1)
        return (image * self.std + self.mean).clamp(0, 1)

    def get_sample(self,
                   pl_module: LightningModule,
                   reals: Tensor | None = None,
                   conds: Dict[str, Tensor] = None):
        fakes = []

        # auto use ema
        with pl_module.ema_scope():
            for _ in range(self.n_ensemble):
                if isinstance(pl_module, VAEModule):
                    recons_img, _ = pl_module.net(reals)
                    fakes.append(recons_img)  # [b, c, w, h]
                elif isinstance(pl_module, DiffusionModule):
                    samples = pl_module.net.sample(
                        num_sample=reals.shape[0],
                        device=pl_module.device,
                        cond=conds.copy() if isinstance(
                            pl_module, ConditionDiffusionModule) else None)
                    fakes.append(samples[-1])  # [b, c, w, h]
                else:
                    raise NotImplementedError('This module is not implemented')

        fakes = torch.stack(fakes, dim=1)  # (b, n, c, w, h)

        return fakes

    def reset_metrics(self):
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

        if self.mean_variance is not None:
            self.mean_variance.reset()

        if self.mean_boundary_variance is not None:
            self.mean_boundary_variance.reset()

    def update_variance(self, fakes: Tensor, device: torch.device):
        # (b, n, c, w, h) -> (b, c, w, h)
        preds = (fakes > 0.5).to(torch.float64)
        variance = preds.var(dim=1)

        if self.mean_variance is not None:
            self.mean_variance.to(device)
            self.mean_variance.update(variance.mean())
            self.mean_variance.to('cpu')

        if self.mean_boundary_variance is not None:
            ids = variance > self.boundary_threshold
            boundary_variance = ((variance * ids).sum(dim=[1, 2, 3]) +
                                 1) / (ids.sum(dim=[1, 2, 3]) + 1)
            self.mean_boundary_variance.to(device)
            self.mean_boundary_variance.update(boundary_variance)
            self.mean_boundary_variance.to('cpu')

    def update_metrics(self, reals: Tensor, fakes: Tensor,
                       device: torch.device):

        fakes = self.rescale(fakes)  # (b, n, c, w, h)
        reals = self.rescale(reals)  # (b, c, w, h)

        if self.mean_variance or self.mean_boundary_variance:
            self.update_variance(fakes, device)

        fakes = fakes.mean(dim=1)

        # update
        if self.ssim is not None:
            self.ssim.to(device)
            self.ssim.update(fakes, reals)
            self.ssim.to('cpu')

        if self.psnr is not None:
            self.psnr.to(device)
            self.psnr.update(fakes, reals)
            self.psnr.to('cpu')

        if self.dice is not None or self.iou is not None:
            preds = (reals > 0.5).to(torch.int64)
            targets = (fakes > 0.5).to(torch.int64)

            if self.dice is not None:
                self.dice.to(device)
                self.dice.update(preds, targets)
                self.dice.to('cpu')

            if self.iou is not None:
                self.iou.to(device)
                self.iou.update(preds, targets)
                self.iou.to('cpu')

        if self.fid is not None or self.IS is not None:
            # gray to rgb image
            if reals.shape[1] == 1:
                reals = torch.cat([reals, reals, reals], dim=1)
                fakes = torch.cat([fakes, fakes, fakes], dim=1)

            reals = torch.nn.functional.interpolate(reals,
                                                    size=(299, 299),
                                                    mode='bilinear')
            fakes = torch.nn.functional.interpolate(fakes,
                                                    size=(299, 299),
                                                    mode='bilinear')

            if self.fid is not None:
                self.fid.to(device)
                self.fid.update(reals, real=True)
                self.fid.update(fakes, real=False)
                self.fid.to('cpu')

            if self.IS is not None:
                self.IS.to(device)
                self.IS.update(fakes)
                self.IS.to('cpu')

    def log_metrics(self, pl_module: LightningModule, mode: str):

        if self.mean_variance is not None:
            self.mean_variance.to(pl_module.device)
            pl_module.log(mode + '/mean_variance',
                          self.mean_variance.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.mean_variance.to('cpu')

        if self.mean_boundary_variance is not None:
            self.mean_boundary_variance.to(pl_module.device)
            pl_module.log(mode + '/mean_boundary_variance',
                          self.mean_boundary_variance.compute(),
                          on_step=False,
                          on_epoch=True,
                          prog_bar=False,
                          sync_dist=True)
            self.mean_boundary_variance.to('cpu')

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
