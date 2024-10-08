from typing import Any, Tuple, List, Dict

import torch
from torch import Tensor
import pyrootutils
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics import MeanMetric
from torch.optim import Optimizer, lr_scheduler
from contextlib import contextmanager

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.diffusion.net import DiffusionModel
from src.utils.ema import LitEma


class DiffusionModule(pl.LightningModule):

    def __init__(
        self,
        net: DiffusionModel,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        use_ema: bool = False,
        compile: bool = False,
    ) -> None:
        
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # diffusion model
        self.net = net

        # loss function
        self.criterion = nn.MSELoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # exponential moving average
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.net)

    def on_train_batch_end(self, *args, **kwargs):
        self.model_ema(self.net)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def forward(self,
                x: Tensor,
                cond: Dict[str, Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: Two tensor of noise
        """
        return self.net(x, cond=cond)

    @torch.no_grad()
    def predict(
        self,
        cond: Dict[str, Tensor] | None = None,
        xt: Tensor | None = None,
        sample_steps: Tensor | None = None,
        num_sample: int = 1,
        noise: Tensor | None = None,
        repeat_noise: bool = False,
        device: torch.device = torch.device('cpu'),
        prog_bar: bool = False,
        get_all_denoise_images: bool = False,
    ) -> List[Tensor]:
        # return list of denoising images

        if self.use_ema:
            with self.ema_scope():
                gen_samples = self.net.sample(cond, xt, sample_steps, num_sample, noise,
                                              repeat_noise, device, prog_bar, get_all_denoise_images)
        else:
            gen_samples = self.net.sample(cond, xt, sample_steps, num_sample, noise,
                                          repeat_noise, device, prog_bar, get_all_denoise_images)

        return gen_samples # range [-1, 1]

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
        self, 
        batch: Tuple[Tensor, Tensor],
    ) -> Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
        """

        batch, _ = batch
        preds, targets = self.forward(batch)
        loss = self.criterion(preds, targets)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)

        self.log("train/loss",
                 self.train_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[Tensor, Tensor],
                        batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)

        self.log("val/loss",
                 self.val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss",
                 self.test_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=self.hparams.scheduler.schedule)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "diffusion")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="diffusion_module.yaml")
    def main(cfg: DictConfig):
        cfg["net"]["n_train_steps"] = 1000
        cfg["net"]["img_dims"] = [1, 32, 32]
        cfg["net"]["sampler"]["n_train_steps"] = 1000
        cfg["net"]["denoise_net"]["in_channels"] = 1
        cfg["net"]["denoise_net"]["out_channels"] = 1
        print(cfg)

        diffusion_module: DiffusionModule = hydra.utils.instantiate(cfg)

        x = torch.randn(2, 1, 32, 32)
        pred, target = diffusion_module(x)
        loss = diffusion_module.model_step(batch=[x, None])

        print('*' * 20, ' DIFFUSION MODULE ', '*' * 20)
        print('Input:', x.shape)
        print('Prediction:', pred.shape)
        print('Target:', target.shape)
        print(f"{diffusion_module.criterion._get_name()}:", loss)

    main()