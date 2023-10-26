from typing import Tuple, List

import torch
from torch import Tensor
import pyrootutils
import torch.nn as nn
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.unet import UNet
from src.models.diffusion.sampler import BaseSampler


class DiffusionModel(nn.Module):
    """
    ### Diffusion Model
    """

    def __init__(
        self,
        denoise_net: UNet,
        sampler: BaseSampler,
        n_train_steps: int = 1000,
        img_dims: Tuple[int, int, int] = [1, 32, 32],
        gif_frequency: int = 20,
    ) -> None:
        """
        denoise_net: model to learn noise
        sampler: sample image in diffusion 
        n_train_steps: the number of  diffusion step for forward process
        img_dims: resolution of image - [channels, width, height]
        gif_frequency: 
        """

        super().__init__()

        self.n_train_steps = n_train_steps
        self.img_dims = img_dims
        self.denoise_net = denoise_net
        self.sampler = sampler
        self.gif_frequency = gif_frequency

    def forward(
        self,
        x0: Tensor,
        sample_steps: Tensor | None = None,
        noise: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        ### forward diffusion process to create label for model training
        x0
        sample_steps:
        noise:
        cond:
        """
        if sample_steps is None:
            #  generate sample timesteps ~ U(0, T - 1)
            sample_steps = torch.randint(
                0,
                self.n_train_steps,
                [x0.shape[0]],
                device=x0.device,
            )
        else:
            assert sample_steps.shape[0] == x0.shape[0], 'batch_size not match'
            sample_steps = sample_steps.to(x0.device)

        if noise is None:
            # noise ~ N(0, I)
            noise = torch.randn_like(x0)
        else:
            assert noise.shape == x0.shape, 'shape not match'
            noise = noise.to(x0.device)

        # diffusion forward: add noise to origin image
        xt = self.sampler.step(x0, sample_steps, noise)

        # noise prediction
        noise_pred = self.denoise_net(x=xt, time_steps=sample_steps, cond=cond)

        return noise_pred, noise

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
        """
        ### reverse diffusion process
        """

        # xt ~ N(0, I)
        if xt is None:
            xt = torch.randn(num_sample,
                             self.img_dims[0],
                             self.img_dims[1],
                             self.img_dims[2],
                             device=device)
        else:
            assert xt.shape[1:] == self.img_dims, 'shape of image not match'
            xt = xt.to(device)

        # list image to generate gif image
        gen_samples = [xt]

        sample_steps = tqdm(
            self.sampler.timesteps,
            desc="Sampling t") if prog_bar else self.sampler.timesteps

        for i, t in enumerate(sample_steps):
            t = torch.full((xt.shape[0], ),
                           t,
                           device=device,
                           dtype=torch.int64)
            model_output = self.denoise_net(x=xt, time_steps=t, cond=cond)
            xt = self.sampler.reverse_step(model_output, t, xt, noise,
                                           repeat_noise)

            if i % self.gif_frequency == 0 or i + 1 == len(
                    self.sampler.timesteps):
                gen_samples.append(xt)

        return gen_samples


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    config_path = str(root / "configs" / "model" / "diffusion" / "net")
    print("root: ", root)

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="diffusion_model.yaml")
    def main(cfg: DictConfig):
        cfg['n_train_steps'] = 1000
        cfg['img_dims'] = [1, 32, 32]
        cfg['sampler']['n_train_steps'] = 1000

        print(cfg)

        diffusion_model: DiffusionModel = hydra.utils.instantiate(cfg)

        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, cfg['n_train_steps'], (2, ))

        print('*' * 20, ' DIFFUSION MODEL ', '*' * 20)

        print('=' * 15, ' forward process ', '=' * 15)
        print('Input:', x.shape)
        pred, target = diffusion_model(x, t)  # with given t
        pred, target = diffusion_model(x)  # without given t
        print('Prediction:', pred.shape)
        print('Target:', target.shape)

        print('=' * 15, ' reverse process ', '=' * 15)
        gen_samples = diffusion_model.sample(num_sample=2, prog_bar=True)
        print(len(gen_samples), gen_samples[0].shape)

    main()