from typing import Any, Tuple, Dict

import torch
from torch import Tensor
import pyrootutils
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torch.optim import Optimizer, lr_scheduler
from contextlib import contextmanager

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.ema import LitEma

class NFModule(pl.LightningModule):

    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        use_ema: bool = False,
        compile: bool = False,
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # flow model
        self.net = net

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

    def forward(self, x: Tensor) -> Dict[Tensor, Tensor]:
        """Perform a forward pass through the model `self.net`."""
        return self.net(x)

    @torch.no_grad()
    def predict(self, num_sample: int = 1, device: torch.device = torch.device("cpu")):
        if self.use_ema:
            with self.ema_scope():
                samples = self.net.sample(num_sample, device)
        else:
            samples = self.net.sample(num_sample, device)
        
        return samples # range [-1, 1]

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
        """
        batch, _ = batch
        _, nll = self.forward(batch)
        return nll

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
    config_path = str(root / "configs" / "model" / "flow")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="nf_module.yaml")
    def main(cfg: DictConfig):
        print(cfg)

        nf_module: NFModule = hydra.utils.instantiate(cfg)

        image = nf_module.predict(num_sample=2)
        _, likelihood = nf_module(image)

        print('*' * 20, ' NF Module ', '*' * 20)
        print(image.shape)
        print("Likelihood", likelihood)

    main()