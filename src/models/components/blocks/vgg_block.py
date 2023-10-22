import torch
import torch.nn as nn
from torch import Tensor


class VGGBlock(nn.Module):
    """
    ### VGG Block
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int | None = None,
                 drop_rate: float = 0.,
                 d_t_emb: int | None = None):
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
                nn.Linear(in_features=d_t_emb, out_features=out_channels),
            )

        # Normalization and convolution in input layer
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
        )

        # Output layers
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
        )

    def forward(self, x: Tensor, t_emb: Tensor | None = None) -> Tensor:
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
        return self.out_layers(out)


if __name__ == "__main__":
    x = torch.randn(2, 32, 10, 10)
    t = torch.randn(2, 32)
    vggBlock_timeEmbedding = VGGBlock(
        in_channels=32,
        out_channels=64,
        d_t_emb=32,
    )
    out1 = vggBlock_timeEmbedding(x, t)
    print('***** VGGBlock_with_TimeEmbedding ***** ')
    print('Input:', x.shape, t.shape)
    print('Output:', out1.shape)

    print('-' * 60)

    vggBlock = VGGBlock(
        in_channels=32,
        out_channels=64,
    )
    out2 = vggBlock(x)
    print('***** VGGBlock_without_TimeEmbedding *****')
    print('Input:', x.shape)
    print('Output:', out2.shape)