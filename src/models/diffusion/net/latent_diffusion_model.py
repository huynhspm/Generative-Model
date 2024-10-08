from typing import List, Tuple, Dict

import torch
from torch import Tensor
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.unet.net import UNetDiffusion
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
        denoise_net: UNetDiffusion,
        sampler: BaseSampler,
        n_train_steps: int = 1000,
        img_dims: Tuple[int, int, int] = [1, 32, 32],
        latent_scaling_factor: float = 1.0,
        classifier_free: bool = False,
    ) -> None:
        """_summary_

        Args:
            autoencoder_weight_path (str): _description_
            denoise_net (UNetDiffusion): model to learn noise
            sampler (BaseSampler): sampler for process with image in diffusion 
            n_train_steps (int, optional): the number of  diffusion step for forward process. Defaults to 1000.
            img_dims (Tuple[int, int, int], optional): resolution of image - [channels, width, height]. Defaults to [1, 32, 32].
            latent_scaling_factor (float, optional): _description_. Defaults to 1.0.
            classifier_free (bool, optional): _description_. Defaults to False.
        """

        super().__init__(denoise_net, sampler, n_train_steps, img_dims, classifier_free)
        assert autoencoder_weight_path is not None, "autoencoder_weight_path must not be None"
        print("autoencoder_weight_path: ", autoencoder_weight_path)

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

    def forward(
        self,
        x0: Tensor,
        sample_steps: Tensor | None = None,
        noise: Tensor | None = None,
        cond: Dict[str, Tensor] | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """_summary_
        ### forward latent diffusion process to create label for model training
        Args:
            x0 (Tensor): _description_
            sample_steps (Tensor | None, optional): _description_. Defaults to None.
            noise (Tensor | None, optional): _description_. Defaults to None.
            cond (Dict[str, Tensor] | None, optional): _description_. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: _description_
        """
        z = self.autoencoder_encode(x0)
        return super().forward(z, sample_steps, noise, cond)

    @torch.no_grad()
    def sample(
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
        """_summary_
        ### reverse latent diffusion process
        Args:
            cond (Dict[str, Tensor], optional): _description_. Defaults to None.
            xt (Tensor | None, optional): _description_. Defaults to None.
            sample_steps (Tensor | None, optional): _description_. Defaults to None.
            num_sample (int, optional): _description_. Defaults to 1.
            noise (Tensor | None, optional): _description_. Defaults to None.
            repeat_noise (bool, optional): _description_. Defaults to False.
            device (torch.device, optional): _description_. Defaults to torch.device('cpu').
            prog_bar (bool, optional): _description_. Defaults to False.
            get_all_denoise_images (bool, optional): _description_. Defaults to False.

        Returns:
            List[Tensor]: _description_
        """
        
        z_samples = super().sample(cond, xt, sample_steps, num_sample, noise,
                                   repeat_noise, device, prog_bar, get_all_denoise_images)
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
        cfg['n_train_steps'] = 1000
        cfg['sampler']['n_train_steps'] = 1000
        # print(cfg)

        latent_diffusion_model: LatentDiffusionModel = hydra.utils.instantiate(
            cfg)

    main()
