from typing import List, Optional, Tuple

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
        cond_net,
        sampler: BaseSampler,
        n_train_steps: int = 1000,
        img_dims: Tuple[int, int, int] = [1, 32, 32],
        gif_frequency: int = 20,
    ) -> None:
        """
        denoise_net: model to learn noise
        cond_net:
        sampler: sample image in diffusion 
        n_train_steps: the number of  diffusion step for forward process
        img_dims: resolution of image - [channels, width, height]
        gif_frequency:
        """

        super().__init__(denoise_net, sampler, n_train_steps, img_dims,
                         gif_frequency)
        self.cond_net = cond_net

    def get_condition_embedding(self, cond: torch.Tensor):
        return self.cond_net(cond)

    def forward(self,
                x0: Tensor,
                sample_steps: Tensor | None = None,
                noise: Tensor | None = None,
                cond: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        if self.cond_net is not None:
            cond = self.get_condition_embedding(cond)
        return super().forward(x0, sample_steps, noise, cond)

    @torch.no_grad()
    def sample(self,
               xt: Tensor | None = None,
               sample_steps: Tensor | None = None,
               cond: Tensor | None = None,
               num_sample: int | None = 1,
               noise: Tensor | None = None,
               repeat_noise: bool = False,
               device: torch.device = torch.device('cpu'),
               prog_bar: bool = False) -> List[Tensor]:
        if self.cond_net is not None:
            cond = self.get_condition_embedding(cond)
        return super().sample(xt, sample_steps, cond, num_sample, noise, repeat_noise,
                              device, prog_bar)


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
        cfg['n_steps'] = 100
        cfg['img_dims'] = [1, 32, 32]
        cfg['denoise_net']['d_cond'] = 256
        cfg['cond_net']['n_classes'] = 2
        print(cfg)

        condition_diffusion_model: ConditionDiffusionModel = hydra.utils.instantiate(
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
