from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 drop_rate: float = 0.,
                 d_t_emb: Optional[int] = None,
                 residual: bool = False) -> None:
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
                nn.Linear(in_features=d_t_emb, out_features=out_channels),
            )

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(),
            nn.Dropout(p=drop_rate),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
        )

    def forward(self,
                x: torch.Tensor,
                t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: is the input feature map with shape `[batch_size, channels, height, width]`
        t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`. defaults to None
        """
        # add skip connection
        if self.residual:
            out = F.silu(x + self.double_conv(x))
        else:
            out = self.double_conv(x)

        #
        if t_emb is not None:
            t_emb = self.emb_layers(t_emb).type(out.dtype)
            return out + t_emb[:, :, None, None]
        else:
            return out


if __name__ == "__main__":
    x = torch.randn(2, 32, 10, 10)
    t = torch.randn(2, 32)
    baseBlock_timeEmbedding = BaseBlock(
        in_channels=32,
        out_channels=64,
        d_t_emb=32,
    )
    out1 = baseBlock_timeEmbedding(x, t)
    print('***** BaseBlock_with_TimeEmbedding *****')
    print('Input:', x.shape, t.shape)
    print('Output:', out1.shape)
    
    print('-' * 60)

    baseBlock = BaseBlock(
        in_channels=32,
        out_channels=64,
    )
    out2 = baseBlock(x)
    print('***** BaseBlock_without_TimeEmbedding *****')
    print('Input:', x.shape)
    print('Output:', out2.shape)