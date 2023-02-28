import math
import torch
import torch.nn as nn
from typing import Tuple, List
from torch.optim import Adam
from pytorch_lightning import LightningModule

from models.components.denoise_models import UnetModel


class DiffusionModel(LightningModule):
    """
    ### Diffusion Model
    """

    def __init__(self,
                 t_range: int,
                 img_dims: Tuple[int, int, int],
                 channels: int = 64,
                 backbone: str = "Base",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4, 4],
                 attention: str = "Base",
                 attention_levels: List[int] = [0, 1, 2],
                 n_attention_heads: int = 4,
                 n_attention_layers: int = 1):
        """
        t_range: the number of diffusion step
        img_dims: the resolution of image
        channels: the base channel count for the model
        backbone: name of block backbone for each level
        n_layer_blocks: number of blocks at each level
        channel_multipliers: the multiplicative factors for number of channels for each level
        attention: name of attentions
        attention_levels: the levels at which attention should be performed
        n_attention_heads: the number of attention heads
        n_attention_layers: the number of attention layers
        """
        super().__init__()
        self.beta_small = 1e-4
        self.beta_large = 0.02
        self.channels = channels
        self.size = img_dims[1]
        self.t_range = t_range

        self.model = UnetModel(img_channels=img_dims[0],
                               channels=self.channels,
                               backbone=backbone,
                               n_layer_blocks=n_layer_blocks,
                               channel_multipliers=channel_multipliers,
                               attention=attention,
                               attention_levels=attention_levels,
                               n_attention_heads=n_attention_heads,
                               n_attention_layers=n_attention_layers)

        self.save_hyperparameters()

    def time_step_embedding(self,
                            time_steps: torch.Tensor,
                            max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings
        time_steps: are the time steps of shape `[batch_size]`
        max_period: controls the minimum frequency of the embeddings.
        """

        half = self.channels // 2
        frequencies = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) /
            half).to(device=time_steps.device)

        args = time_steps[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        result = 1
        for j in range(t):
            result *= self.alpha(j)
        return result

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (self.beta_large -
                                                       self.beta_small)

    def forward(self, x, time_steps):
        # reshape time steps to [batch_size]
        time_steps = time_steps.squeeze(1)

        t_emb = self.time_step_embedding(time_steps)

        output = self.model(x, t_emb)
        return output

    def get_loss(self, batch, batch_idx):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        ts = torch.randint(0,
                           self.t_range, [batch.shape[0]],
                           device=self.device)
        noise_imgs = []
        epsilons = torch.randn(batch.shape, device=self.device)

        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])  # alpha bar t
            noise_imgs.append((math.sqrt(a_hat) * batch[i]) +
                              (math.sqrt(1 - a_hat) * epsilons[i])  # x_t
                              )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        # noise prediction
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))

        in_size = self.size * self.size
        # loss = nn.functional.mse_loss(e_hat.reshape(-1, in_size),
        #                               epsilons.reshape(-1, in_size))

        loss = nn.functional.mse_loss(e_hat.reshape(batch.shape[0], -1),
                                      epsilons.reshape(batch.shape[0], -1))
        return loss

    def denoise_sample(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape, device=self.device)
            else:
                z = 0
            e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1))
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=2e-4)
        return optimizer


if __name__ == "__main__":
    input = torch.randn(2, 3, 64, 64)
    df = DiffusionModel(10, [3, 64, 64])

    output = df.forward(input, torch.Tensor([[1], [2]]))
    print(output.shape)