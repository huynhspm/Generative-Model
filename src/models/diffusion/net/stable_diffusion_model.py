from typing import List, Tuple, Dict

import torch
from torch import Tensor
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.unet import UNet
from src.models.vae import VAEModule
from src.models.diffusion.sampler import BaseSampler
from src.models.diffusion.net import ConditionDiffusionModel


class StableDiffusionModel(ConditionDiffusionModel):
    """
    ### Stable Diffusion Model
    """

    def __init__(
        self,
        denoise_net: UNet,
        sampler: BaseSampler,
        autoencoder_weight_path: str,
        image_embedder: torch.nn.Module = None,
        text_embedder: torch.nn.Module = None,
        n_train_steps: int = 1000,
        img_dims: Tuple[int, int, int] = [1, 32, 32],
        gif_frequency: int = 20,
        latent_scaling_factor: float = 1.0,
    ) -> None:
        """_summary_

        Args:
            autoencoder_weight_path (str): _description_
            denoise_net (UNet): model to learn noise
            sampler (BaseSampler): mampler for process with image in diffusion
            image_embedder (torch.nn.Module, optional): _description_. Defaults to None.
            text_embedder (torch.nn.Module, optional): _description_. Defaults to None.
            n_train_steps (int, optional): the number of  diffusion step for forward process. Defaults to 1000.
            img_dims (Tuple[int, int, int], optional): resolution of image - [channels, width, height]. Defaults to [1, 32, 32].
            gif_frequency (int, optional): _description_. Defaults to 20.
            latent_scaling_factor (float, optional): _description_. Defaults to 1.0.
        """

        super().__init__(denoise_net, image_embedder, text_embedder, sampler,
                         n_train_steps, img_dims, gif_frequency)
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
                cond: Dict[str, Tensor] = None) -> Tuple[Tensor, Tensor]:
        """_summary_
        ### forward diffusion process to create label for model training
        Args:
            x0 (Tensor): _description_
            sample_steps (Tensor | None, optional): _description_. Defaults to None.
            noise (Tensor | None, optional): _description_. Defaults to None.
            cond (Dict[str, Tensor], optional): _description_. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]:
                - pred: noise is predicted from xt by model
                - target: noise is added to (x0 -> xt)
        """
        z = self.autoencoder_encode(x0)
        return super().forward(z, sample_steps, noise, cond)

    @torch.no_grad()
    def sample(self,
               xt: Tensor | None = None,
               sample_steps: Tensor | None = None,
               cond: Dict[str, Tensor] = None,
               num_sample: int = 1,
               noise: Tensor | None = None,
               repeat_noise: bool = False,
               device: torch.device = torch.device('cpu'),
               prog_bar: bool = False) -> List[Tensor]:
        """_summary_
        ### reverse diffusion process
        Args:
            xt (Tensor | None, optional): _description_. Defaults to None.
            sample_steps (Tensor | None, optional): _description_. Defaults to None.
            cond (Dict[str, Tensor], optional): _description_. Defaults to None.
            num_sample (int | None, optional): _description_. Defaults to 1.
            noise (Tensor | None, optional): _description_. Defaults to None.
            repeat_noise (bool, optional): _description_. Defaults to False.
            device (torch.device, optional): _description_. Defaults to torch.device('cpu').
            prog_bar (bool, optional): _description_. Defaults to False.

        Returns:
            List[Tensor]: _description_
        """

        z_samples = super().sample(xt, sample_steps, cond, num_sample, noise,
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
                config_name="stable_diffusion_model.yaml")
    def main(cfg: DictConfig):
        cfg['n_steps'] = 100
        cfg['img_dims'] = [1, 32, 32]
        cfg['denoise_net']['d_cond'] = 256
        cfg['cond_net']['n_classes'] = 2
        print(cfg)

        condition_diffusion_model: StableDiffusionModel = hydra.utils.instantiate(
            cfg)

        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 100, (2, ))
        cond = torch.randint(0, 2, (2, ))

        print('***** q_sample *****')
        print('Input:', x.shape)
        targets, preds = condition_diffusion_model.get_q_sample(x, cond=cond)
        print('Output:', targets.shape, preds.shape)

        print('-' * 60)

        print('***** p_sample *****')
        t = Tensor([2]).to(torch.int64)
        images = condition_diffusion_model.get_p_sample(num_sample=2,
                                                        cond=cond,
                                                        prog_bar=True)
        print(len(images), images[0].shape)
        # print(latent_diffusion_model.denoise_sample(x, t).shape)

        print('-' * 60)

        out = condition_diffusion_model(
            x, t, cond=condition_diffusion_model.get_condition_embedding(cond))
        print('***** Condition_Diffusion_Model *****')
        print('Input:', x.shape)
        print('Output:', out.shape)

    main()
