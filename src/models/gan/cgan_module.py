from typing import Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.gan import GANModule
from src.models.gan.net import CGAN


class CGANModule(GANModule):

    def __init__(self,
                net: CGAN,
                optimizer_gen: Optimizer,
                optimizer_disc: Optimizer,
                use_ema: bool = False,
                compile: bool = False) -> None:
        super().__init__(net, optimizer_gen, optimizer_disc, use_ema, compile)

    def training_step(self, batch: Tuple[Tensor, Tensor],
                    batch_idx: int) -> Tensor:

        images, cond = batch
        opt_gen, opt_disc  = self.optimizers()

        gen_loss = self.get_gen_loss(cond, num_images=len(images))
        self.update_params(opt_gen, gen_loss)

        disc_loss = self.get_disc_loss(cond, real=images)
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


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "gan")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="cgan_module.yaml")
    def main(cfg: DictConfig):
        cfg["net"]["gen"]["latent_dim"] = 100
        cfg["net"]["gen"]["d_cond_label"] = 10
        cfg["net"]["disc"]["img_dims"] = [1, 32, 32]
        cfg["net"]["disc"]["d_cond_label"] = 10
        print(cfg)

        cgan_module: CGANModule = hydra.utils.instantiate(cfg)
        
        cond={"label": torch.tensor([0, 1], dtype=torch.int64)}
        image = cgan_module.predict(cond=cond, 
                                    num_sample=2)

        gen_loss = cgan_module.get_gen_loss(cond=cond, num_images=2)
        disc_loss = cgan_module.get_disc_loss(cond=cond, real=image)

        print('*' * 20, ' CGAN Module ', '*' * 20)
        print(image.shape)
        print("Gen-Loss", gen_loss)
        print("Disc-Loss", disc_loss)
        print('-' * 100)

    main()