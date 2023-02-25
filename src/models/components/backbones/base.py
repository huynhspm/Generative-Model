import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 drop_rate: float = 0.,
                 d_t_emb: int = None,
                 residual: bool = False):
        """
        in_channels: the number of input channels
        out_channels: is the number of out channels. defaults to `in_channels.
        drop_rate: parameter of dropout layer
        d_t_emb: the size of timestep embeddings if not None. defaults to None
        """
        super().__init__()
        self.residual = residual

        # `out_channels` not specified
        if out_channels is None:
            out_channels = in_channels

        if d_t_emb is not None:
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
            nn.Dropout(drop_rate),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None):
        """
        x: is the input feature map with shape `[batch_size, channels, height, width]`
        t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`. defaults to None
        """
        # add skip connection
        if self.residual:
            out = F.gelu(x + self.double_conv(x))
        else:
            out = self.double_conv(x)

        #
        if t_emb is not None:
            t_emb = self.emb_layers(t_emb).type(out.dtype)
            return out + t_emb[:, :, None, None]
        else:
            return out


if __name__ == "__main__":
    x = torch.randn(1, 32, 10, 10)
    t = torch.randn(1, 32)
    baseBlock1 = BaseBlock(
        in_channels=32,
        out_channels=64,
        d_t_emb=32,
    )
    out1 = baseBlock1(x, t)
    print(out1.shape)

    baseBlock2 = BaseBlock(
        in_channels=32,
        out_channels=64,
    )
    out2 = baseBlock2(x)

    print(out2.shape)