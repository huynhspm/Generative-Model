import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, channels: int, n_heads: int):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor):
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value


class SAWrapper(nn.Module):

    def __init__(self,
                 channels: int,
                 n_heads: int,
                 n_layers: int,
                 d_cond: int = None):
        super(SAWrapper, self).__init__()
        self.sa = nn.Sequential(
            *[SelfAttention(channels, n_heads) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        batch_size, channels, size, _ = x.shape
        x = x.view(batch_size, channels, -1).swapaxes(1, 2)
        x = self.sa(x)
        x = x.swapaxes(2, 1).view(batch_size, channels, size, size)
        return x


if __name__ == "__main__":
    input = torch.randn(1, 256, 16, 16)
    sa = SAWrapper(256, 4, 1)
    output = sa(input)
    print(output.shape)