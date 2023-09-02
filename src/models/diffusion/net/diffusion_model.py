from typing import Tuple, Optional, List

import math
import torch
from torch import Tensor
import pyrootutils
import torch.nn as nn
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.unet import UNet

class DiffusionModel(nn.Module):
    """
    ### Diffusion Model
    """

    def __init__(
        self,
        denoise_net: UNet,
        n_steps: int = 1000,
        img_dims: Tuple[int, int, int] = [1, 32, 32],
        schedule_noise: str = 'base',
    ) -> None:
        """
        n_steps: the number of diffusion step
        img_dims: resolution of image - [channels, width, height]
        denoise_net: learning noise
        """

        super().__init__()

        self.n_steps = n_steps
        self.img_dims = img_dims
        self.denoise_net = denoise_net

        self.beta = self.get_beta_schedule(schedule_noise=schedule_noise)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def get_beta_schedule(self,
                          schedule_noise: str,
                          beta_start: float = 1e-4,
                          beta_end: float = 0.02,
                          max_beta: float = 0.999,
                          s: float = 0.008):

        def sigmoid(x):
            return 1 / (torch.exp(-x) + 1)

        if schedule_noise == 'base':
            beta = Tensor([
                beta_start + (t / self.n_steps) * (beta_end - beta_start)
                for t in range(self.n_steps)
            ])
        elif schedule_noise == "linear":
            beta = torch.linspace(beta_start, beta_end, self.n_steps)
        elif schedule_noise == "cosine":
            f = [
                math.cos((t / self.n_steps + s) / (1 + s) * (math.pi / 2))**2
                for t in range(self.n_steps + 1)
            ]

            beta = []
            for t in range(self.n_steps):
                beta.append((min(1 - f[t + 1] / f[t], max_beta)))
            return Tensor(beta)

        elif schedule_noise == "quad":
            beta = (torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                self.n_steps,
                dtype=torch.float64,
            )**2)
        elif schedule_noise == "const":
            beta = beta_end * torch.ones(self.n_steps, dtype=torch.float64)
        elif schedule_noise == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            beta = 1.0 / torch.linspace(
                self.n_steps + 1, 0, self.n_steps, dtype=torch.float64)
        elif schedule_noise == "sigmoid":
            beta = torch.linspace(-6, 6, self.n_steps)
            beta = sigmoid(beta) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(
                f"unknown beta schedule: {schedule_noise}")

        assert beta.shape == (self.n_steps, )

        return beta

    def gather(self, const: Tensor, t: Tensor) -> Tensor:
        """
        ### Gather const for t and reshape to feature map shape
        """
        c = const.to(t.device).gather(-1, t)
        return c.reshape(-1, 1, 1, 1)

    def q_sample(self,
                 x0: Tensor,
                 t: Tensor,
                 eps: Optional[Tensor] = None) -> Tensor:
        """
        ### Sample from q(x_t|x_0). x_t is sampled from x0 by adding noise
        """

        # epsilon ~ N(0, I)
        if eps is None:
            eps = torch.randn_like(x0, device=x0.device)

        alpha_bar = self.gather(self.alpha_bar, t)
        mean = alpha_bar**0.5 * x0
        var = 1 - alpha_bar

        return mean + (var**0.5) * eps

    def get_q_sample(
        self,
        x0: Tensor,
        sample_steps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        ### foward diffusion process
        """

        if sample_steps is None:
            #  generate sample timesteps ~ U(0, T - 1)
            sample_steps = torch.randint(
                0,
                self.n_steps,
                [x0.shape[0]],
                device=x0.device,
            )

        if noise is None:
            # noise ~ N(0, I)
            noise = torch.randn_like(x0, device=x0.device)

        # diffusion forward: add noise to origin image
        xt = self.q_sample(x0, sample_steps, eps=noise)
        
        # noise prediction
        eps_theta = self.forward(xt, sample_steps, cond)

        return eps_theta.reshape(
            -1, self.img_dims[1] * self.img_dims[2]), noise.reshape(
                -1, self.img_dims[1] * self.img_dims[2])
                
    def forward(self,
                x: Tensor,
                time_steps: Tensor,
                cond: Optional[Tensor] = None) -> Tensor:
        # noise prediction
        eps_theta = self.denoise_net(x, time_steps, cond)
        return eps_theta

    @torch.no_grad()
    def p_sample(
        self,
        xt: Tensor,
        t: Tensor,
        t_prev: Tensor = None,
        eta: float = 0,
        eps: Optional[Tensor] = None,
        repeat_noise: bool = False,
        cond: Optional[Tensor] = None,
    ):
        """
        eta: control the sampling stochasticity. 0 for DDIM and 1 for DDPM
        """

        # noise prediction
        eps_theta = self.forward(xt, t.repeat(xt.shape[0]), cond)

        alpha_bar = self.gather(self.alpha_bar, t)

        # compute x0
        x0_pred = (xt - (1 - alpha_bar)**0.5 * eps_theta) / (alpha_bar**0.5)

        if t_prev == 0: return x0_pred

        alpha_bar_prev = self.gather(self.alpha_bar, t_prev)

        # eps ~ N(0, I)
        if eps is None:
            if repeat_noise:
                eps = torch.randn(1, xt.shape[1:], device=xt.device)
            else:
                eps = torch.randn(xt.shape, device=xt.device)

        var = (1 - alpha_bar_prev) / (1 - alpha_bar) * self.gather(
            self.beta, t)
        # control the sampling stochasticity
        var = var * eta

        mean = alpha_bar_prev**0.5 * x0_pred + (1 - alpha_bar_prev -
                                                var)**0.5 * eps_theta

        return mean + (var**0.5) * eps

    def get_timesteps_sample(self,
                             discretize: str = 'uniform',
                             n_steps: int = 50,
                             device: str = 'cpu'):
        if discretize == 'uniform':
            skip = self.n_steps // n_steps
            time_steps = torch.arange(0, self.n_steps, skip, device=device)
        elif discretize == 'quad':
            time_steps = (torch.linspace(0, (self.n_steps * .8)**0.5,
                                         n_steps,
                                         device=device))**2
        else:
            raise NotImplementedError(f"unknown discretize: {discretize}")

        return time_steps

    def get_p_sample(self,
                     xt: Optional[Tensor] = None,
                     sample_steps: Optional[Tensor] = None,
                     cond: Optional[Tensor] = None,
                     num_sample: int = 1,
                     gen_type: str = 'ddim',
                     repeat_noise: bool = False,
                     discretize: str = 'uniform',
                     device: str = 'cpu',
                     prog_bar: bool = False) -> List[Tensor]:
        """
        ### reverse diffusion process
        """

        # generate xt ~ N(0, I)
        if xt is None:
            xt = torch.randn(num_sample,
                             self.img_dims[0],
                             self.img_dims[1],
                             self.img_dims[2],
                             device=device)
        
        # list image to generate gif image
        gen_samples = []

        if sample_steps is not None:
            sample_steps = sample_steps.to(device)
            if prog_bar:
                sample_steps = tqdm(sample_steps, total=len(sample_steps))
            for t in sample_steps:
                xt = self.denoise_sample(xt, t)
                if t % 50 == 0: gen_samples.append(xt)
            return gen_samples

        # base or ddpm or ddim
        if gen_type == 'base':
            # original sample
            sample_steps = torch.arange(self.n_steps-1, -1, -1, device=device)
            if prog_bar:
                sample_steps = tqdm(sample_steps, total=len(sample_steps))
            for t in sample_steps:
                xt = self.denoise_sample(xt, t)
                if t % 50 == 0: gen_samples.append(xt)
            return gen_samples
        
        elif gen_type == 'ddpm':
            n_steps = self.n_steps
            eta = 1.0
        elif gen_type == 'ddim':
            n_steps = int(self.n_steps / 20)
            eta = 0.0
        else:
            NotImplementedError(f"unknown generate type: {gen_type}")

        # generate sample timesteps
        sample_steps = self.get_timesteps_sample(discretize=discretize,
                                                 n_steps=n_steps,
                                                 device=device)

        # reverse sample_steps to backward
        sample_steps = sample_steps.flip(dims=[0])

        if prog_bar:
            sample_steps = tqdm(enumerate(zip(sample_steps[:-1], sample_steps[1:])),
                     total=len(sample_steps) - 1)
        else:
            sample_steps = enumerate(zip(sample_steps[:-1], sample_steps[1:]))

        for _, (t, t_prev) in sample_steps:
            xt = self.p_sample(xt=xt,
                               t=t,
                               t_prev=t_prev,
                               eta=eta,
                               repeat_noise=repeat_noise,
                               cond=cond)
            if t_prev % 20 == 0:
                gen_samples.append(xt)
        return gen_samples

    def denoise_sample(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape, device=x.device)
            else:
                z = 0
            e_hat = self.forward(x, t.repeat(x.shape[0]))
            pre_scale = 1 / math.sqrt(self.gather(self.alpha, t))
            e_scale = (1 - self.gather(
                self.alpha, t)) / math.sqrt(1 - self.gather(self.alpha_bar, t))
            post_sigma = math.sqrt(self.gather(self.beta, t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x


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
        cfg['n_steps'] = 100
        cfg['img_dims'] = [1, 32, 32]

        # print(cfg)
                    
        diffusion_model: DiffusionModel = hydra.utils.instantiate(cfg)

        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 100, (2, ))
        out1 = diffusion_model(x, t)
        print('***** Diffusion_Model *****')
        print('Input:', x.shape)
        print('Output:', out1.shape)

        print('-' * 60)
        
        print('***** q_sample *****')
        print('Input:', x.shape)
        targets, preds = diffusion_model.get_q_sample(x)
        print('Output:', targets.shape, preds.shape)

        print('-' * 60)

        print('***** p_sample *****')
        t = Tensor([2]).to(torch.int64)
        cond = Tensor([[1], [2]]).to(torch.int64)
        images = diffusion_model.get_p_sample(num_sample=2, prog_bar=True)
        print(len(images), images[0].shape)
        print(diffusion_model.denoise_sample(x, t).shape)
        
        print('-' * 60)

        cfg.denoise_net.n_classes = 2 
        cond = torch.randint(0, 2, (2, ))
        cond_diffusion : DiffusionModel = hydra.utils.instantiate(cfg)
        out2 = cond_diffusion(x, t, cond=cond)
        print('***** Condition_Diffusion_Model *****')
        print('Input:', x.shape)
        print('Output:', out2.shape)

    main()
