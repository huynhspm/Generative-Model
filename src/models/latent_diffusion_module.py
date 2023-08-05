from typing import Any

import torch
import pyrootutils
from torch.optim import Optimizer, lr_scheduler

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models import DiffusionModule
from src.models.components import DiffusionModel
from src.models.components.autoencoder import AutoEncoder


class LatentDiffusionModule(DiffusionModule):

    def __init__(
        self,
        net: DiffusionModel,
        autoencoder: AutoEncoder,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
    ) -> None:
        super().__init__(net, optimizer, scheduler)

        # autoencoder
        self.autoencoder = autoencoder

    def model_step(self, batch: Any):
        batch, _ = batch
        batch_encoded = self.autoencoder_encode(batch)
        preds, targets = self.forward(batch_encoded)
        loss = self.criterion(preds, targets)
        return loss, preds, targets
    
    def autoencoder_encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        ### Get scaled latent space representation of the image
        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """
        return self.autoencoder.encode(image).sample()

    def autoencoder_decode(self, z: torch.Tensor):
        """
        ### Get image from the latent representation
        We scale down by the scaling factor and then decode.
        """
        return self.autoencoder.decode(z)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="latent_diffusion_module.yaml")
    def main(cfg: DictConfig):
        latent_diffusion_module: LatentDiffusionModule = hydra.utils.instantiate(cfg)
        input = torch.randn(2, 1, 32, 32)
        preds, targets = latent_diffusion_module(input)
        print(preds.shape, targets.shape)

    main()