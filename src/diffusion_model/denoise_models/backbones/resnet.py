import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(self,
                 in_channels: int,
                 d_t_emb: int,
                 out_channels: int = None):
        """
        in_channels: the number of input channels
        d_t_emb: the size of timestep embeddings
        out_channels: is the number of out channels. defaults to `in_channels.
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = in_channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        # Time step embeddings
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels),
        )

        # Final convolution layer
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1))

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        # Initial convolution
        h = self.in_layers(x)
        # Time step embeddings
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        # Add time step embeddings
        h = h + t_emb[:, :, None, None]
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h