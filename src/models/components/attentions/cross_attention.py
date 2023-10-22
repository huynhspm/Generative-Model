import torch
from torch import Tensor
import torch.nn as nn


class CrossAttention(nn.Module):

    def __init__(self, channels: int, n_heads: int):
        super(CrossAttention, self).__init__()
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
        if cond is None:
            cond = x_ln

        attention_value, _ = self.mha(x_ln, cond, cond)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value


class CAWrapper(nn.Module):

    def __init__(self,
                 channels: int,
                 n_heads: int,
                 n_layers: int,
                 d_cond: int | None = None):
        super(CAWrapper, self).__init__()
        self.ca_layers = nn.ModuleList(
            [CrossAttention(channels, n_heads) for _ in range(n_layers)])

        if d_cond is not None:
            self.linear = nn.Linear(d_cond, channels)

    def forward(self, x: Tensor, cond: Tensor | None = None):
        batch_size, channels, size, _ = x.shape
        x = x.view(batch_size, channels, -1).swapaxes(1, 2)

        if cond is not None:
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
    ca = CAWrapper(256, 4, 1, 128)
    output = ca(input, cond)
    print(output.shape)
