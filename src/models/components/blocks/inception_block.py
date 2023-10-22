import torch
import torch.nn as nn
from torch import Tensor


class InceptionBlock(nn.Module):
    """
    ### Inception Block
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int | None = None,
                 drop_rate: float = 0.,
                 d_t_emb: int | None = None) -> None:
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
                nn.Linear(in_features=d_t_emb, out_features=in_channels),
            )

        # Normalization and convolution in input layer
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=3,
                      padding=1),
        )

        # inception
        out_channels //= 4

        # Normalization and convolution in branch 1
        self.branch_1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
        )

        ## Normalization and convolution in branch 2
        self.branch_2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
        )

        # Normalization and convolution in branch 3
        self.branch_3 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1),
        )

        # Normalization and convolution in branch 4
        self.branch_4 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1),
        )

    def forward(self, x: Tensor, t_emb: Tensor = None) -> Tensor:
        """
        x: is the input feature map with shape `[batch_size, channels, height, width]`
        t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`. defaults to None
        """

        # Normalization and convolution in input layer
        out = self.in_layers(x)

        if t_emb is not None:
            # Time step embeddings
            t_emb = self.emb_layers(t_emb).type(out.dtype)
            # Add time step embeddings
            out = out + t_emb[:, :, None, None]

        # inception

        # Normalization and convolution in branch 1
        branch_3x3dbl = self.branch_1(out)

        # Normalization and convolution in branch 2
        branch_3x3 = self.branch_2(out)

        # Normalization and convolution in branch 3
        branch_pool = self.branch_3(out)

        # Normalization and convolution in branch 4
        branch_1x1 = self.branch_4(out)

        # concat in inception_block
        return torch.cat((branch_3x3dbl, branch_3x3, branch_pool, branch_1x1),
                         dim=1)


if __name__ == "__main__":
    x = torch.randn(1, 32, 10, 10)
    t = torch.randn(1, 32)
    inceptionBlock_timeEmbedding = InceptionBlock(
        in_channels=32,
        out_channels=64,
        d_t_emb=32,
    )
    out1 = inceptionBlock_timeEmbedding(x, t)
    print('***** InceptionBlock_with_TimeEmbedding *****')
    print('Input:', x.shape, t.shape)
    print('Output:', out1.shape)

    print('-' * 60)

    inceptionBlock = InceptionBlock(
        in_channels=32,
        out_channels=64,
    )
    out2 = inceptionBlock(x)
    print('***** InceptionBlock_without_TimeEmbedding *****')
    print('Input:', x.shape)
    print('Output:', out2.shape)