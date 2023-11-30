from typing import Tuple, List, Dict

import torch
from torch import Tensor
import pyrootutils
import torch.nn as nn
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.unet import UNet
from src.models.diffusion.sampler import BaseSampler
from src.models.diffusion.sampler import noise_like


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
        """_summary_

        Args:
            denoise_net (UNet): model to learn noise
            sampler (BaseSampler): sampler for process with image in diffusion
            n_train_steps (int, optional): the number of  diffusion step for forward process. Defaults to 1000.
            img_dims (Tuple[int, int, int], optional): resolution of image - [channels, width, height]. Defaults to [1, 32, 32].
            gif_frequency (int, optional): _description_. Defaults to 20.
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
        cond: Dict[str, Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
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

        # xt ~ N(0, I)
        if xt is None:
            xt = noise_like([num_sample] + list(self.img_dims),
                            device=device,
                            repeat=repeat_noise)
        else:
            assert xt.shape[1:] == self.img_dims, 'shape of image not match'
            xt = xt.to(device)

        # list image to generate gif image
        gen_samples = [xt]

        if sample_steps is None:
            sample_steps = tqdm(
                self.sampler.timesteps) if prog_bar else self.sampler.timesteps
        else:
            assert sample_steps.shape[0] == xt.shape[
                0], 'batch of sample_steps and xt not match'
            sample_steps = tqdm(sample_steps) if prog_bar else sample_steps

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
        # print(cfg)

        diffusion_model: DiffusionModel = hydra.utils.instantiate(cfg)

        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, cfg['n_train_steps'], (2, ))

        print('*' * 20, ' DIFFUSION MODEL ', '*' * 20)

        print('=' * 15, ' forward process ', '=' * 15)
        print('Input:', x.shape)
        xt = diffusion_model.sampler.step(x, t)
        pred, target = diffusion_model(x, )  # with given t
        pred, target = diffusion_model(x)  # without given t
        print('xt:', xt.shape)
        print('Prediction:', pred.shape)
        print('Target:', target.shape)

        print('=' * 15, ' reverse process ', '=' * 15)
        gen_samples = diffusion_model.sample(num_sample=2, prog_bar=True)
        print(len(gen_samples), gen_samples[0].shape)

    main()