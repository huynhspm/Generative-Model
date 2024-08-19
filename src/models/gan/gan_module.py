from typing import Any, Tuple, Dict

import torch
from torch import Tensor
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torch.optim import Optimizer
from contextlib import contextmanager
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.gan.net import GAN
from src.utils.ema import LitEma


class GANModule(pl.LightningModule):

    def __init__(self,
                net: GAN,
                optimizer_gen: Optimizer,
                optimizer_disc: Optimizer,
                use_ema: bool = False,
                compile: bool = False) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # for manual backward
        self.automatic_optimization = False

        # diffusion model
        self.net = net

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # for averaging loss across batches
        self.train_gen_loss = MeanMetric()
        self.train_disc_loss = MeanMetric()

        self.val_gen_loss = MeanMetric()
        self.val_disc_loss = MeanMetric()

        self.test_gen_loss = MeanMetric()
        self.test_disc_loss = MeanMetric()


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

    def predict(self, 
                cond: Dict[str, Tensor],
                num_sample: int = 1,
                device: torch.device = torch.device("cpu")) -> Tensor:
        # return generating samples

        if self.use_ema:
            with self.ema_scope():
                samples = self.net.sample(cond, num_sample, device)
        else:
            samples = self.net.sample(cond, num_sample, device)
        
        return samples # range [-1, 1]
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def get_disc_loss(self, cond: Dict[str, Tensor], real: Tensor) -> Tensor:
        # generate a batch (num_images) of fake images.
        fake = self.net.sample(cond=cond, num_sample=len(real), device=self.device)

        # Get the discriminator's prediction of the fake image and calculate the loss.
        fake_pred = self.net.classify(cond=cond, image=fake.detach())
        
        # Get the discriminator's prediction of the real image and calculate the loss.
        real_pred = self.net.classify(cond=cond, image=real)

        # Calculate the discriminator's loss by averaging the real 
        # and fake loss and set it to disc_loss.
        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
        disc_loss = (fake_loss + real_loss) / 2
        
        return disc_loss

    def get_gen_loss(self, cond: Dict[str, Tensor], num_images: Tensor) -> Tensor:
        # Create noise vectors and generate a batch of fake images. 
        fake = self.net.sample(cond=cond, num_sample=num_images, device=self.device)

        # Get the discriminator's prediction of the fake image.
        fake_pred = self.net.classify(cond=cond, image=fake)

        # Calculate the generator's loss.
        gen_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))

        return gen_loss

    def update_params(self, optimizer: Optimizer, loss: Tensor):
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
    
    def training_step(self, batch: Tuple[Tensor, Tensor],
                    batch_idx: int) -> Tensor:

        images, _ = batch
        opt_gen, opt_disc  = self.optimizers()

        gen_loss = self.get_gen_loss(cond=None, num_images=len(images))
        self.update_params(opt_gen, gen_loss)

        disc_loss = self.get_disc_loss(cond=None, real=images)
        self.update_params(opt_disc, disc_loss)

        # update and log metrics
        self.train_gen_loss(gen_loss)
        self.train_disc_loss(disc_loss)

        self.log("train/disc_loss",
                self.train_disc_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True)
    
        self.log("train/gen_loss",
                self.train_gen_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True)

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
        images, _ = batch

        gen_loss = self.get_gen_loss(cond=None, num_images=len(images))
        disc_loss = self.get_disc_loss(cond=None, real=images)

        # update and log metrics
        self.val_gen_loss(gen_loss)
        self.val_disc_loss(disc_loss)

        self.log("val/disc_loss",
                self.val_disc_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True)
    
        self.log("val/gen_loss",
                self.val_gen_loss,
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
        images, _ = batch

        gen_loss = self.get_gen_loss(cond=None, num_images=len(images))
        disc_loss = self.get_disc_loss(cond=None, real=images)

        # update and log metrics
        self.test_gen_loss(gen_loss)
        self.test_disc_loss(disc_loss)

        self.log("test/disc_loss",
                self.test_disc_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True)
    
        self.log("test/gen_loss",
                self.test_gen_loss,
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
        optimizer_gen = self.hparams.optimizer_gen(params=self.net.gen.parameters())
        optimizer_disc = self.hparams.optimizer_gen(params=self.net.disc.parameters())

        return [{"optimizer": optimizer_gen}, {"optimizer": optimizer_disc}]


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "gan")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="gan_module.yaml")
    def main1(cfg: DictConfig):
        cfg["net"]["gen"]["latent_dim"] = 100
        cfg["net"]["disc"]["img_dims"] = [1, 32, 32]
        print(cfg)

        gan_module: GANModule = hydra.utils.instantiate(cfg)
        image = gan_module.predict(cond=None, num_sample=2)

        gen_loss = gan_module.get_gen_loss(cond=None, num_images=2)
        disc_loss = gan_module.get_disc_loss(cond=None, real=image)

        print('*' * 20, ' GAN Module ', '*' * 20)
        print(image.shape)
        print("Gen-Loss", gen_loss)
        print("Disc-Loss", disc_loss)
        print('-' * 100)

    @hydra.main(version_base=None,
            config_path=config_path,
            config_name="dcgan_module.yaml")
    def main2(cfg: DictConfig):
        cfg["net"]["disc"]["img_channels"] = 1
        cfg["net"]["disc"]["img_size"] = 32
        print(cfg)

        dcgan_module: GANModule = hydra.utils.instantiate(cfg)
        image = dcgan_module.predict(cond=None, num_sample=2)

        disc_loss = dcgan_module.get_disc_loss(cond=None, real=image)
        gen_loss = dcgan_module.get_gen_loss(cond=None, num_images=2)

        print('*' * 20, ' DCGAN Module ', '*' * 20)
        print(image.shape)
        print("Gen-Loss", gen_loss)
        print("Disc-Loss", disc_loss)

    main1()
    main2()