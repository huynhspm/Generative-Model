from typing import List, Optional, Tuple

import torch
from torch import Tensor
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.unet import UNet
from src.models.vae import VAEModule
from src.models.diffusion.net import DiffusionModel
from src.models.diffusion.sampler import BaseSampler


class LatentDiffusionModel(DiffusionModel):
    """
    ### Latent Diffusion Model
    """

    def __init__(
            self,
            autoencoder_weight_path: str,
            denoise_net: UNet,
            sampler: BaseSampler,
            n_train_steps: int = 1000,
            img_dims: Tuple[int, int, int] = [1, 32, 32],
            gif_frequency: int = 20,
            latent_scaling_factor: float = 1.0,  # 0.3643
    ) -> None:
        """
        autoencoder_weight_path
        denoise_net: model to learn noise
        sampler: sample image in diffusion 
        n_train_steps: the number of  diffusion step for forward process
        img_dims: resolution of image - [channels, width, height]
        gif_frequency:
        latent_scaling_factor:
        """

        super().__init__(denoise_net, sampler, n_train_steps, img_dims,
                         gif_frequency)
        assert autoencoder_weight_path is not None, "autoencoder_weight_path must not be None"
        self.autoencoder_module: VAEModule = VAEModule.load_from_checkpoint(
            autoencoder_weight_path)
        self.autoencoder_module.eval().freeze()
        self.latent_scaling_factor = latent_scaling_factor

    @torch.no_grad()
    def autoencoder_encode(self, image: Tensor) -> Tensor:
        """
        ### Get scaled latent space representation of the image
        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """
        z, _ = self.autoencoder_module.net.encode(image)
        return z * self.latent_scaling_factor

    @torch.no_grad()
    def autoencoder_decode(self, z: Tensor):
        """
        ### Get image from the latent representation
        We scale down by the scaling factor and then decode.
        """
        return self.autoencoder_module.net.decode(z /
                                                  self.latent_scaling_factor)

    def forward(self,
                x0: Tensor,
                sample_steps: Tensor | None = None,
                noise: Tensor | None = None,
                cond: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        z = self.autoencoder_encode(x0)
        return super().forward(z, sample_steps, noise, cond)

    @torch.no_grad()
    def sample(self,
               xt: Tensor | None = None,
               sample_steps: Tensor | None = None,
               cond: Tensor | None = None,
               num_sample: int | None = 1,
               repeat_noise: bool = False,
               device: torch.device = torch.device('cpu'),
               prog_bar: bool = False) -> List[Tensor]:

        z_samples = super().sample(xt, sample_steps, cond, num_sample,
                                   repeat_noise, device, prog_bar)
        return [self.autoencoder_decode(z) for z in z_samples]


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    config_path = str(root / "configs" / "model" / "diffusion" / "net")
    print("root: ", root)

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="latent_diffusion_model.yaml")
    def main(cfg: DictConfig):
        cfg['n_steps'] = 100

        # print(cfg)

        latent_diffusion_model: LatentDiffusionModel = hydra.utils.instantiate(
            cfg)

        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 100, (2, ))
        out1 = latent_diffusion_model(x, t)
        print('***** Latent Diffusion_Model *****')
        print('Input:', x.shape)
        print('Output:', out1.shape)

        print('-' * 60)

        print('***** q_sample *****')
        print('Input:', x.shape)
        targets, preds = latent_diffusion_model.get_q_sample(x)
        print('Output:', targets.shape, preds.shape)

        print('-' * 60)

        print('***** p_sample *****')
        t = Tensor([2]).to(torch.int64)
        cond = Tensor([[1], [2]]).to(torch.int64)
        images = latent_diffusion_model.get_p_sample(num_sample=2,
                                                     prog_bar=True)
        print(len(images), images[0].shape)
        # print(latent_diffusion_model.denoise_sample(x, t).shape)

        print('-' * 60)

        cfg.denoise_net.n_classes = 2
        cond = torch.randint(0, 2, (2, ))
        x = torch.randn(2, 32, 8, 8)
        cond_latent_diffusion: LatentDiffusionModel = hydra.utils.instantiate(
            cfg)
        out2 = cond_latent_diffusion(x, t, cond=cond)
        print('***** Condition_Diffusion_Model *****')
        print('Input:', x.shape)
        print('Output:', out2.shape)

    main()
