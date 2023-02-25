import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """
    ### Inception Block
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 drop_rate: float = 0.,
                 d_t_emb: int = None) -> None:
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
            # Time step embedding
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_t_emb, in_channels),
            )

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

        # inception
        out_channels //= 4
        self.branch_1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.branch_2 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.branch_3 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, 1),
        )

        self.branch_4 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None):
        """
        x: is the input feature map with shape `[batch_size, channels, height, width]`
        t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`. defaults to None
        """

        # First normalization and convolution
        out = self.in_layers(x)

        if t_emb is not None:
            # Time step embeddings
            t_emb = self.emb_layers(t_emb).type(out.dtype)
            # Add time step embeddings
            out = out + t_emb[:, :, None, None]

        # inception
        branch_3x3dbl = self.branch_1(out)
        branch_3x3 = self.branch_2(out)
        branch_pool = self.branch_3(out)
        branch_1x1 = self.branch_4(out)

        # mix
        out = torch.cat((branch_3x3dbl, branch_3x3, branch_pool, branch_1x1),
                        dim=1)

        #
        return out


if __name__ == "__main__":
    x = torch.randn(1, 32, 10, 10)
    t = torch.randn(1, 32)
    inceptionBlock1 = InceptionBlock(
        in_channels=32,
        out_channels=64,
        d_t_emb=32,
    )
    out1 = inceptionBlock1(x, t)
    print(out1.shape)

    inceptionBlock2 = InceptionBlock(
        in_channels=32,
        out_channels=64,
    )
    out2 = inceptionBlock2(x)

    print(out2.shape)