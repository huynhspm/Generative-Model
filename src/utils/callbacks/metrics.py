from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import Tensor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio, FrechetInceptionDistance,
                                InceptionScore)

from src.models.diffusion import DiffusionModule, ConditionDiffusionModule
from src.models.vae import VAEModule


class Metrics(Callback):

    def __init__(self,
                 ssim: StructuralSimilarityIndexMeasure | None = None,
                 psnr: PeakSignalNoiseRatio | None = None,
                 fid: FrechetInceptionDistance | None = None,
                 IS: InceptionScore | None = None,
                 mean: float = 0.5,
                 std: float = 0.5):
        self.ssim = ssim
        self.psnr = psnr
        self.fid = fid
        self.IS = IS

        self.mean = mean
        self.std = std

    # def on_train_start(self, trainer: Trainer,
    #                    pl_module: LightningModule) -> None:
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

    def on_validation_start(self, trainer: Trainer,
                            pl_module: LightningModule) -> None:
        self.reset_metrics()

    def on_validation_batch_end(self, trainer: Trainer,
                                pl_module: LightningModule,
                                outputs: STEP_OUTPUT | None, batch: Any,
                                batch_idx: int, dataloader_idx: int) -> None:
        reals, conds = batch
        fakes = self.get_sample(pl_module, reals, conds)
        self.update_metrics(reals, fakes, device=pl_module.device)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.log_metrics(pl_module, mode='val')

    def on_test_start(self, trainer: Trainer,
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
                   conds: Tensor | None = None):
        if isinstance(pl_module, VAEModule):
            if pl_module.use_ema:
                with pl_module.ema_scope():
                    fakes = pl_module.net(reals)[0]
            else:
                fakes = pl_module.net(reals)[0]
        elif isinstance(pl_module, DiffusionModule):
            if not isinstance(pl_module, ConditionDiffusionModule):
                conds = None

            if pl_module.use_ema:
                with pl_module.ema_scope():
                    samples = pl_module.net.sample(num_sample=reals.shape[0],
                                                   device=pl_module.device,
                                                   cond=conds)
            else:
                samples = pl_module.net.sample(num_sample=reals.shape[0],
                                               device=pl_module.device,
                                               cond=conds)
            fakes = samples[-1]
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
            self.fid.to('cpu')
