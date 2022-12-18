import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 d_t_emb: int,
                 out_channels: int = None,
                 residual: bool = False):
        """
        in_channels: the number of input channels
        d_t_emb: the size of timestep embeddings
        out_channels: is the number of out channels. defaults to `channels.
    
        """
        super().__init__()
        self.residual = residual

        if not out_channels:
            out_channels = in_channels

        # Time step embeddings
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels),
        )

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        if self.residual:
            out = F.gelu(x + self.double_conv(x))
        else:
            out = self.double_conv(x)

        t_emb = self.emb_layers(t_emb).type(out.dtype)

        return out + t_emb[:, :, None, None]