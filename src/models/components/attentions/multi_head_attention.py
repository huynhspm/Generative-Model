import torch
from torch import Tensor
import torch.nn as nn


class MHA(nn.Module):
    """_summary_
    Multi-Head-Attention
    """

    def __init__(self, channels: int, n_heads: int):
        super(MHA, self).__init__()
        self.mha = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: Tensor, cond: Tensor | None = None):
        """
        x: shape [b, c, w*h]
        """
        x_ln = self.ln(x)

        # self_attention if cond is None else cross_attention
        cond = x_ln if cond is None else cond

        attention_value, _ = self.mha(x_ln, cond, cond)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value


class SABlock(nn.Module):
    """_summary_
    Self-Attention-Block
    """

    def __init__(self,
                 channels: int,
                 n_heads: int,
                 n_layers: int,
                 d_cond=None):
        super(SABlock, self).__init__()
        self.sa = nn.Sequential(
            *[MHA(channels, n_heads) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, cond=None):
        batch_size, channels, size, _ = x.shape
        x = x.view(batch_size, channels, -1).swapaxes(1, 2)
        x = self.sa(x)
        x = x.swapaxes(2, 1).view(batch_size, channels, size, size)
        # because: Warning: Grad strides do not match bucket view strides
        x = x.contiguous()
        return x


#
class CABlock(nn.Module):
    """_summary_
    Cross-Attention-Block
    """

    def __init__(self, channels: int, n_heads: int, n_layers: int,
                 d_cond: int):
        super(CABlock, self).__init__()
        self.ca_layers = nn.ModuleList(
            [MHA(channels, n_heads) for _ in range(n_layers)])

        self.d_cond = d_cond
        self.linear = nn.Linear(d_cond, channels)

    def forward(self, x: Tensor, cond: Tensor):
        batch_size, channels, size, _ = x.shape
        x = x.view(batch_size, channels, -1).swapaxes(1, 2)

        cond = self.linear(cond)

        for ca in self.ca_layers:
            x = ca(x, cond)

        x = x.swapaxes(2, 1).view(batch_size, channels, size, size)

        # because: Warning: Grad strides do not match bucket view strides
        x = x.contiguous()

        return x


if __name__ == "__main__":
    input = torch.randn(2, 256, 16, 16)
    cond = torch.randn(2, 1, 128)
    print('Input:', input.shape)
    print('Condition:', cond.shape)

    print('*' * 20, ' SELF ATTENTION ', '*' * 20)
    sa = SABlock(channels=256, n_heads=4, n_layers=1)
    output = sa(input, cond)
    print('Output:', output.shape)

    print('*' * 20, ' CROSS ATTENTION ', '*' * 20)
    ca = CABlock(256, 4, 1, 128)
    ca = CABlock(channels=256, n_heads=4, n_layers=1, d_cond=128)
    output = ca(input, cond)
    print('Output:', output.shape)
