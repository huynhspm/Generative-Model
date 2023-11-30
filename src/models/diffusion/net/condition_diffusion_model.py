from typing import List, Tuple, Dict

import torch
from torch import Tensor
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.unet import UNet
from src.models.diffusion.net import DiffusionModel
from src.models.diffusion.sampler import BaseSampler


class ConditionDiffusionModel(DiffusionModel):
    """
    ### Condition Diffusion Model
    """

    def __init__(
        self,
        denoise_net: UNet,
        sampler: BaseSampler,
        image_embedder: torch.nn.Module = None,
        text_embedder: torch.nn.Module = None,
        n_train_steps: int = 1000,
        img_dims: Tuple[int, int, int] = [1, 32, 32],
        gif_frequency: int = 20,
    ) -> None:
        """_summary_
        
        Args:
            denoise_net (UNet): model to learn noise
            sampler (BaseSampler): mampler for process with image in diffusion
            image_embedder (torch.nn.Module, optional): _description_. Defaults to None.
            text_embedder (torch.nn.Module, optional): _description_. Defaults to None.
            n_train_steps (int, optional): the number of  diffusion step for forward process. Defaults to 1000.
            img_dims (Tuple[int, int, int], optional): resolution of image - [channels, width, height]. Defaults to [1, 32, 32].
            gif_frequency (int, optional): _description_. Defaults to 20.
        """
        super().__init__(denoise_net, sampler, n_train_steps, img_dims,
                         gif_frequency)
        self.image_embedder = image_embedder
        self.text_embedder = text_embedder

    def get_image_embedding(self, image: torch.Tensor):
        return self.image_embedder(image)

    def get_text_embedding(self, text: torch.Tensor):
        return self.text_embedder(text)

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

        if cond is not None:
            if 'image' in cond.keys() and self.image_embedder is not None:
                cond['image'] = self.get_image_embedding(cond['image'])

            if 'text' in cond.keys():
                assert ('text' in cond.keys()) == (self.text_embedder
                                                   is not None)
                cond['text'] = self.get_text_embedding(cond['text'])

        return super().forward(x0, sample_steps, noise, cond)

    @torch.no_grad()
    def sample(self,
               xt: Tensor | None = None,
               sample_steps: Tensor | None = None,
               cond: Dict[str, Tensor] = None,
               num_sample: int | None = 1,
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

        if cond is not None:
            if 'image' in cond.keys() and self.image_embedder is not None:
                cond['image'] = self.get_image_embedding(cond['image'])

            if 'text' in cond.keys():
                assert ('text' in cond.keys()) == (self.text_embedder
                                                   is not None)
                cond['text'] = self.get_text_embedding(cond['text'])

        return super().sample(xt, sample_steps, cond, num_sample, noise,
                              repeat_noise, device, prog_bar)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    config_path = str(root / "configs" / "model" / "diffusion" / "net")
    print("root: ", root)

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="condition_diffusion_model.yaml")
    def main(cfg: DictConfig):
        cfg['n_train_steps'] = 1000
        cfg['img_dims'] = [1, 32, 32]
        cfg['sampler']['n_train_steps'] = 1000
        cfg['denoise_net']['d_cond_image'] = 1
        cfg['denoise_net']['n_classes'] = 2
        # print(cfg)

        condition_diffusion_model: ConditionDiffusionModel = hydra.utils.instantiate(
            cfg)

        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, cfg['n_train_steps'], (2, ))
        cond = {
            'label': torch.randint(0, cfg['denoise_net']['n_classes'], (2, )),
            'image': torch.rand_like(x),
        }

        print('***** CONDITION DIFFUSION MODEL *****')

        print('=' * 15, ' forward process ', '=' * 15)
        print('Input:', x.shape)
        xt = condition_diffusion_model.sampler.step(x, t)
        pred, target = condition_diffusion_model(x, cond=cond)
        pred, target = condition_diffusion_model(x, cond=cond)
        print('xt:', xt.shape)
        print('Prediction:', pred.shape)
        print('Target:', target.shape)

        print('=' * 15, ' reverse process ', '=' * 15)
        gen_samples = condition_diffusion_model.sample(num_sample=2,
                                                       cond=cond,
                                                       prog_bar=True)
        print(len(gen_samples), gen_samples[0].shape)

    main()
