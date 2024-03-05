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
            mean_variance
            mean (float, optional): to convert image into (0, 1). Defaults to 0.5.
            std (float, optional): to convert image into (0, 1). Defaults to 0.5.
            n_ensemble
        """

        self.ssim = ssim
        self.psnr = psnr
        self.fid = fid
        self.IS = IS
        self.dice = dice
        self.iou = iou
        self.mean_variance = mean_variance
        self.mean_boundary_variance = mean_boundary_variance

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

    def get_sample(self,
                   pl_module: LightningModule,
                   reals: Tensor | None = None,
                   conds: Dict[str, Tensor] = None):

        if isinstance(pl_module, VAEModule):
            if pl_module.use_ema:
                with pl_module.ema_scope():
                    fakes = pl_module.net(reals)[0]
            else:
                fakes = pl_module.net(reals)[0]
        elif isinstance(pl_module, DiffusionModule):
            if not isinstance(pl_module, ConditionDiffusionModule):
                conds = None

            # auto use ema
            with pl_module.ema_scope():
                fakes = []
                b, c, w, h = reals.shape
                for _ in range(self.n_ensemble):
                    samples = pl_module.net.sample(num_sample=b,
                                                   device=pl_module.device,
                                                   cond=conds.copy())
                    fakes.append(samples[-1])
            fakes = torch.cat(fakes, dim=0)
            fakes = fakes.reshape(self.n_ensemble, b, c, w, h)
            variance = fakes.var(dim=0)

            if self.mean_variance is not None:
                self.mean_variance.to(pl_module.device)
                self.mean_variance.update(variance.mean())
                self.mean_variance.to('cpu')

            if self.mean_boundary_variance is not None:
                threshold= 0.05
                ids = variance > threshold
                boundary_variance = (variance * ids).sum(
                    dim=[1, 2, 3]) / ids.sum(dim=[1, 2, 3])
                self.mean_boundary_variance.to(pl_module.device)
                self.mean_boundary_variance.update(boundary_variance)
                self.mean_boundary_variance.to('cpu')

            fakes = fakes.mean(dim=0)

        else:
            raise NotImplementedError('this module is not Implemented')

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

    def update_metrics(self, reals: Tensor, fakes: Tensor,
                       device: torch.device):
        # convert range (-1, 1) to (0, 1)
        fakes = (fakes * self.std + self.mean).clamp(0, 1)
        reals = (reals * self.std + self.mean).clamp(0, 1)

        # update
        if self.ssim is not None:
            self.ssim.to(device)
            self.ssim.update(fakes, reals)
            self.ssim.to('cpu')

        if self.psnr is not None:
            self.psnr.to(device)
            self.psnr.update(fakes, reals)
            self.psnr.to('cpu')

        targets = (reals > 0.5).to(torch.int64)

        if self.dice is not None:
            self.dice.to(device)
            self.dice.update(fakes, targets)
            self.dice.to('cpu')

        if self.iou is not None:
            self.iou.to(device)
            self.iou.update(fakes, targets)
            self.iou.to('cpu')

        # gray image
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
