import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    """
    ### VGG Block
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 drop_rate: float = 0.,
                 d_t_emb: int = None):
        """
        in_channels: the number of input channels
        out_channels: is the number of out channels. defaults to `in_channels.
        drop_rate: parameter of dropout layer
        d_t_emb: the size of timestep embeddings if not None. defaults to None
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = in_channels

        if d_t_emb is not None:
            # Time step embeddings
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_t_emb, out_channels),
            )

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        # Output layers
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None):
        """
        x: is the input feature map with shape `[batch_size, channels, height, width]`
        t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`. defaults to None
        """
        # Initial convolution
        out = self.in_layers(x)

        if t_emb is not None:
            # Time step embeddings
            t_emb = self.emb_layers(t_emb).type(x.dtype)
            # Add time step embeddings
            out = out + t_emb[:, :, None, None]

        # Output layers
        out = self.out_layers(out)

        #
        return out


if __name__ == "__main__":
    x = torch.randn(1, 32, 10, 10)
    t = torch.randn(1, 32)
    vggBlock1 = VGGBlock(
        in_channels=32,
        out_channels=64,
        d_t_emb=32,
    )
    out1 = vggBlock1(x, t)
    print(out1.shape)

    vggBlock2 = VGGBlock(
        in_channels=32,
        out_channels=64,
    )
    out2 = vggBlock2(x)

    print(out2.shape)