from typing import List, Any, Tuple, Dict

import torch
from torch import Tensor
import pyrootutils
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torch.optim import Optimizer, lr_scheduler
from contextlib import contextmanager

from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from segmentation_models_pytorch.losses import DiceLoss

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vae.net import BaseVAE
from src.utils.ema import LitEma


class VAEModule(pl.LightningModule):

    def __init__(
        self,
        net: BaseVAE,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        use_ema: bool = False,
        compile: bool = False,
        loss: str = "mse",
        weight_loss: List[int] = None,
    ):
        """_summary_

        Args:
            net (BaseVAE): _description_
            optimizer (Optimizer): _description_
            scheduler (lr_scheduler): _description_
            use_ema (bool, optional): _description_. Defaults to False.
            loss (str, optional): loss function for reconstruction. Defaults to "mse".
            weight_loss (_type_, optional): for weighted-cross-entropy-loss. Defaults to List[int]=None.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # autoencoder
        self.net = net

        # loss function
        if loss == "mse":
            self.criterion = MSELoss()
        elif loss =="ce":
            self.criterion = CrossEntropyLoss(weight=torch.tensor(weight_loss, dtype=torch.float32))
        elif loss == "bce":
            self.criterion = BCEWithLogitsLoss(weight=torch.tensor(weight_loss))
        elif loss == "dice":
            self.criterion = DiceLoss(mode="binary", log_loss=True, from_logits=False)
        else:
            raise NotImplementedError(f"not implemented {loss}-loss")
        
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

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of images
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def compute_loss(self, preds: Tensor, targets:Tensor, loss: Dict[str, Tensor]):
        if isinstance(self.criterion, (BCEWithLogitsLoss, DiceLoss)):
            targets = targets * 0.5 + 0.5
            preds = preds * 0.5 + 0.5
        elif isinstance(self.criterion, CrossEntropyLoss):
            targets = (targets.squeeze(dim=1) * 0.5 + 0.5).to(torch.int64)
            preds = preds * 0.5 + 0.5
            preds = torch.cat((1 - preds, preds), dim=1)

        recons_loss = self.criterion(preds, targets)

        # only for reconstruction -> autoencoder
        if loss is None:
            return {"loss": recons_loss}
        
        # variational autoencoder
        loss["recons_loss"] = recons_loss
        loss["loss"] = sum(loss.values())
        
        for key in loss:
            if key == "loss": continue
            loss[key] = loss[key].detach()
        
        return loss

        
    def model_step(
            self, batch: Tuple[Tensor,
                               Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        batch, _ = batch
        preds, loss = self.forward(batch)
        losses = self.compute_loss(preds, batch, loss)
        return losses

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses = self.model_step(batch)
    
        # update and log metrics
        self.train_loss(losses["loss"])

        self.log("train/loss",
                 self.train_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)


        for key in losses.keys():
            if key == "loss": continue

            self.log(f"train/{key}",
                    losses[key],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True)
            
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": losses["loss"]}

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
        losses = self.model_step(batch)

        # update and log metrics
        self.val_loss(losses['loss'])
        keys = [key for key in losses.keys()]

        self.log("val/loss",
                 self.val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        
        for key in losses.keys():
            if key == "loss": continue

            self.log(f"train/{key}",
                    losses[key],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses = self.model_step(batch)

        # update and log metrics
        self.test_loss(losses['loss'])
        keys = [key for key in losses.keys()]

        self.log("test/loss",
                 self.test_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        
        for key in losses.keys():
            if key == "loss": continue

            self.log(f"train/{key}",
                    losses[key],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True)

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


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "vae")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="vq_vae_module.yaml")
    def main(cfg: DictConfig):

        # cfg['net']['kld_weight']=[0, 1]
        cfg['net']['img_dims'] = [3, 32, 32]
         # print(cfg)


        vae_module: VAEModule = hydra.utils.instantiate(cfg)
        vae: BaseVAE = hydra.utils.instantiate(cfg.get('net'))

        x = torch.randn(2, 3, 32, 32)
        out, loss = vae_module(x)

        print('***** VAE_Module *****')
        print('Input:', x.shape)
        print('Output:', out.shape)
        print('Loss:', loss)
        print(vae_module.model_step([x, None]))

    main()