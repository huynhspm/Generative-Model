import torch
from torch import Tensor
import pyrootutils
import torch.nn as nn

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.diffusion.sampler import BetaSchedule, VarianceType, BaseSampler, expand_dim_like


class DDPMSampler(BaseSampler):

    def __init__(self,
                 n_train_steps: int = 1000,
                 n_infer_steps: int = 1000,
                 beta_schedule: BetaSchedule = 'linear',
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 given_betas: Tensor | None = None,
                 variance_type: VarianceType = "fixed_large",
                 clip_denoised: bool = True,
                 set_final_alpha_to_one: bool = True) -> None:
        """
        n_train_steps:
        n_infer_Steps:
        beta_schedule:
        beta_start:
        beta_end:
        given_betas:
        variance_type:
        clip_denoised:
        set_final_alpha_to_one:
        """

        super().__init__(n_train_steps, n_infer_steps, beta_schedule,
                         beta_start, beta_end, given_betas, variance_type,
                         clip_denoised, set_final_alpha_to_one)

    def reverse_step(self,
                     model_output: Tensor,
                     t: Tensor,
                     xt: Tensor,
                     noise: Tensor | None = None,
                     repeat_noise: bool = False) -> Tensor:

        # ddpm so: t_prev = t - 1
        t_prev = t - self.n_train_steps // self.n_infer_steps

        alpha_bar = self.alphas_bar[t].flatten()
        alpha_bar = expand_dim_like(alpha_bar, xt)

        alpha_bar_prev = torch.where(t_prev >= 0, self.alphas_bar[t_prev],
                                     self.final_alpha_bar).flatten()
        alpha_bar_prev = expand_dim_like(alpha_bar_prev, xt)

        alpha = alpha_bar / alpha_bar_prev

        sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar[t].flatten()
        sqrt_one_minus_alphas_bar = expand_dim_like(sqrt_one_minus_alphas_bar,
                                                    xt)
        if noise is None:
            if repeat_noise:
                noise = torch.randn(1, xt.shape[1:], device=xt.device)
            else:
                noise = torch.randn_like(xt)
        else:
            assert noise.shape == xt.shape, 'shape not match'
            noise = noise.to(xt.device)

        # formula (11) from https://arxiv.org/pdf/2006.11239.pdf
        mean = (xt -
                (1 - alpha) / sqrt_one_minus_alphas_bar * model_output) / (
                    alpha**0.5)

        variance = torch.zeros_like(model_output)
        # if t = 0 (the last step reverse) -> not add noise
        ids = t > 0
        variance[ids] = expand_dim_like(self.get_variance(t, t_prev)[ids], xt)

        std = variance**0.5

        return mean + std * noise


if __name__ == "__main__":
    sampler = DDPMSampler()