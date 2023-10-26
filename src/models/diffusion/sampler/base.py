from typing import Literal, TypeAlias, Union

import math
import torch
import torch.nn as nn
from torch import Tensor

BetaSchedule: TypeAlias = Literal["base", "linear", "cosine", "scaled_linear",
                                  "const", "jsd", "squaredcos_cap_v2",
                                  "sigmoid"]

VarianceType = Literal["fixed_small", "fixed_large", "fixed_large_log"]


def expand_dim_like(x: Tensor, y: Tensor):
    while x.ndim < y.ndim:
        x = x.unsqueeze(-1)
    return x


class BaseSampler(nn.Module):

    def __init__(self,
                 n_train_steps: int,
                 n_infer_steps: int,
                 beta_schedule: BetaSchedule = 'linear',
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 given_betas: Tensor | None = None,
                 variance_type: VarianceType = "fixed_small",
                 clip_denoised: bool = True,
                 set_final_alpha_to_one: bool = True):
        super().__init__()
        self.n_train_steps = n_train_steps
        self.clip_denoised = clip_denoised
        self.variance_type = variance_type

        self.set_timesteps(n_infer_steps)

        self.register_schedule(given_betas,
                               beta_schedule,
                               beta_start,
                               beta_end,
                               set_final_alpha_to_one,
                               n_steps=n_train_steps)

    def register_schedule(
        self,
        given_betas: Tensor | None = None,
        beta_schedule: BetaSchedule = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        set_final_alpha_to_one: bool = True,
        n_steps: int = 1000,
    ):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(n_steps, beta_schedule, beta_start,
                                       beta_end)

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        final_alpha_bar = Tensor(
            [1.0]) if set_final_alpha_to_one else alphas_bar[0]

        # automatic to device
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar",
                             torch.sqrt(1.0 - alphas_bar))
        self.register_buffer("final_alpha_bar", final_alpha_bar)

    def set_timesteps(self, n_infer_steps: int, discretize: str = 'uniform'):
        if n_infer_steps > self.n_train_steps:
            raise ValueError(
                f"Number of inference steps ({n_infer_steps}) cannot be large than number of train steps ({self.n_train_steps})"
            )
        else:
            self.n_infer_steps = n_infer_steps

        if discretize == 'uniform':
            self.timesteps = (torch.linspace(0, self.n_train_steps - 1,
                                             n_infer_steps).flip(0).to(
                                                 torch.int64))
        elif discretize == 'quad':
            self.timesteps = (torch.linspace(0, (self.n_train_steps * .8)**0.5,
                                             n_infer_steps)**2).flip(0).to(
                                                 torch.int64)
        else:
            raise NotImplementedError(f"unknown discretize: {discretize}")

    def step(self,
             x0: Tensor,
             t: Tensor,
             noise: Tensor | None = None) -> Tensor:
        """Perform a single step of the diffusion process.
        Sample from q(x_t|x_0). x_t is sampled from x0 by adding noise

        Args:
            x0: samples with noiseless (Input).
            t: Timesteps in the diffusion chain.
            noise: The noise tensor for the current timestep.

        Returns:
            xt (Tensor): The noisy samples.
        """
        # epsilon ~ N(0, I)
        if noise is None:
            noise = torch.randn_like(x0)
        else:
            assert noise.shape == x0.shape, 'shape not match'
            noise = noise.to(x0.device)

        sqrt_alpha_prod = self.sqrt_alphas_bar[t].flatten()
        sqrt_alpha_prod = expand_dim_like(sqrt_alpha_prod, x0)

        sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar[t].flatten()
        sqrt_one_minus_alphas_bar = expand_dim_like(sqrt_one_minus_alphas_bar,
                                                    x0)

        mean = sqrt_alpha_prod * x0
        std = sqrt_one_minus_alphas_bar

        return mean + std * noise

    def get_variance(self,
                     t: Tensor,
                     t_prev: Tensor,
                     predicted_variance: Tensor | None = None):
        alpha_bar = self.alphas_bar[t]
        alpha_bar_prev = torch.where(t_prev >= 0, self.alphas_bar[t_prev],
                                     self.final_alpha_bar)

        beta = 1 - alpha_bar / alpha_bar_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        variance = (1 - alpha_bar_prev) / (1 - alpha_bar) * beta

        # hacks - were probably added for training stability
        if self.variance_type == "fixed_small":
            variance = torch.clamp(variance, min=1e-20)
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif self.variance_type == "fixed_large":
            variance = beta
        elif self.variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(beta)
        else:
            raise ValueError(f"Unknown variance type {self.variance_type}")

        return variance

    def reverse_step(self,
                     model_output: Tensor,
                     t: Tensor,
                     xt: Tensor,
                     eta: float = 1.0,
                     noise: Tensor | None = None,
                     repeat_noise: bool = False) -> Tensor:
        """Predict the sample at the previous timestep by reversing the SDE.

        Args:
            model_output: The output of the denoise model.
            t: Timesteps in the diffusion chain.
            xt: Current instance of sample being created by diffusion forward process.
            noise:
            repeat_noise:  
        Returns:
            x{t-1}: Samples at the previous timesteps.
        """
        t_prev = t - self.n_train_steps // self.n_infer_steps

        alpha_bar = self.alphas_bar[t].flatten()
        alpha_bar = expand_dim_like(alpha_bar, xt)

        sqrt_one_minus_alphas_bar = self.sqrt_one_minus_alphas_bar[t].flatten()
        sqrt_one_minus_alphas_bar = expand_dim_like(sqrt_one_minus_alphas_bar,
                                                    xt)

        alpha_bar_prev = torch.where(t_prev >= 0, self.alphas_bar[t_prev],
                                     self.final_alpha_bar).flatten()
        alpha_bar_prev = expand_dim_like(alpha_bar_prev, xt)

        # compute x_0 of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        x0_pred = (xt - sqrt_one_minus_alphas_bar * model_output) / (alpha_bar
                                                                     **0.5)

        if self.clip_denoised:
            x0_pred.clamp_(-1.0, 1.0)

        if noise is None:
            if repeat_noise:
                noise = torch.randn(1, xt.shape[1:], device=xt.device)
            else:
                noise = torch.randn_like(xt)
        else:
            assert noise.shape == xt.shape, 'shape not match'
            noise = noise.to(xt.device)

        variance = torch.zeros_like(model_output)
        # if t = 0 (the last step reverse) -> not add noise
        ids = t > 0
        variance[ids] = expand_dim_like(self.get_variance(t, t_prev)[ids], xt)
        std = variance**0.5

        # control the sampling stochasticity
        std = std * eta

        mean = alpha_bar_prev**0.5 * x0_pred + (1 - alpha_bar_prev -
                                                variance)**0.5 * model_output

        return mean + std * noise


def make_beta_schedule(
    n_steps: int = 1000,
    beta_schedule: str = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    max_beta: float = 0.999,
    s: float = 0.008,
    device: Union[torch.device, str] | None = None,
):

    def sigmoid(x):
        return 1 / (torch.exp(-x) + 1)

    if beta_schedule == 'base':
        betas = Tensor([
            beta_start + (t / n_steps) * (beta_end - beta_start)
            for t in range(n_steps)
        ])
    elif beta_schedule == "linear":
        betas = torch.linspace(beta_start,
                               beta_end,
                               n_steps,
                               dtype=torch.float32,
                               device=device)
    elif beta_schedule == "cosine":
        f = [
            math.cos((t / n_steps + s) / (1 + s) * (math.pi / 2))**2
            for t in range(n_steps + 1)
        ]

        betas = []
        for t in range(n_steps):
            betas.append((min(1 - f[t + 1] / f[t], max_beta)))
        return Tensor(betas)

    elif beta_schedule == "scaled_linear":
        # this schedule is very specific to the latent diffusion model
        betas = (torch.linspace(beta_start**0.5,
                                beta_end**0.5,
                                n_steps,
                                dtype=torch.float32,
                                device=device)**2)
    elif beta_schedule == "const":
        betas = beta_end * torch.ones(n_steps, dtype=torch.float32)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / torch.linspace(
            n_steps + 1, 0, n_steps, dtype=torch.float32)
    elif beta_schedule == "squaredcos_cap_v2":
        # Glide cosine schedule
        betas = betas_for_alpha_bar(n_steps, max_beta)
    elif beta_schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_steps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"unknown beta schedule: {beta_schedule}")

    assert betas.shape == (n_steps, ), f'not enough {n_steps} steps'

    return betas


def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2)**2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(
            f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return Tensor(betas, dtype=torch.float32)