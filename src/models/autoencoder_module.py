from typing import List, Any

import torch
import pyrootutils
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression.mae import MeanAbsoluteError
from torch.optim import Optimizer, lr_scheduler

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.autoencoder import AutoEncoder

class AutoEncoderModule(pl.LightningModule):

    def __init__(
        self,
        net: AutoEncoder,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])

        # autoencoder
        self.net = net

        # loss function
        self.criterion = nn.MSELoss()

        # metric objects for calculating and averaging MAE across batches
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_mae_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        self.val_mae_best.reset()

    def model_step(self, batch: Any):
        batch, _ = batch
        preds = self.forward(batch)
        loss = self.criterion(preds, batch)

        return loss, preds, batch

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_mae(preds, targets)

        self.log("train/loss",
                 self.train_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_mae(preds, targets)

        self.log("val/loss",
                 self.val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss }

    def validation_epoch_end(self, outputs: List[Any]):
        mae = self.val_mae.compute()  # get current val mae
        self.val_mae_best(mae)  # update best so far val mae
        # log `val_mae_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_mae(preds, targets)
        
        self.log("test/loss",
                 self.test_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


from typing import Optional
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3",
            config_path="../../configs/model",
            config_name="autoencoder_module.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    cfg.net.decoder.out_channels = 1
    autoencoder_module : AutoEncoderModule = hydra.utils.instantiate(cfg)
    autoencoder : AutoEncoder = hydra.utils.instantiate(cfg.get('net'))
    
    x = torch.randn(2, 1, 32, 32)
    print('input:', x.shape)
    print('output encoder:', autoencoder.encode(x).sample().shape)
    print('output autoencoder', autoencoder(x).shape)
    print('output autoencoder_module', autoencoder_module(x).shape)

if __name__ == "__main__":
    main()