from typing import List

import torch
import pyrootutils
from torch import nn
from torch import Tensor

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.blocks import init_block
from src.models.components.attentions import init_attention
from src.models.components.up_down import DownSample


class Encoder(nn.Module):
    """
    ### Encoder module
    """

    def __init__(self,
                 in_channels: int,
                 z_channels: int = 3,
                 base_channels: int = 64,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4],
                 attention: str = "Attention",
                 double_z: bool = False) -> None:
        """
        in_channels: is the number of channels in the input
        z_channels: is the number of channels in the embedding space
        base_channels: is the number of channels in the first convolution layer
        block: is the block of block in each layers of encoder
        n_layer_blocks: is the number of resnet layers at each resolution
        channel_multipliers: are the multiplicative factors for the number of channels in the subsequent blocks
        attention:
        double_z:
        """
        super().__init__()

        # Number of levels downSample
        levels = len(channel_multipliers)

        # Number of channels at each level
        channels_list = [base_channels * m for m in channel_multipliers]

        channels = base_channels

        # Block to downSample
        Block = init_block(block)

        # attention layer
        Attention = init_attention(attention)

        # Input convolution
        self.encoder_input = nn.Conv2d(in_channels=in_channels,
                                       out_channels=channels,
                                       kernel_size=3,
                                       padding=1)

        # List of top-level blocks
        self.encoder = nn.ModuleList()

        # Prepare layer for downSampling
        for i in range(levels):
            # Add the blocks, attentions and downSample
            blocks = nn.ModuleList()

            for _ in range(n_layer_blocks):
                blocks.append(
                    Block(
                        in_channels=channels,
                        out_channels=channels_list[i],
                    ))

                channels = channels_list[i]

            down = nn.Module()
            down.blocks = blocks

            # Down-sampling at the end of each top level block except the last
            if i != levels - 1:
                down.downSample = DownSample(channels=channels)
            else:
                down.downSample = nn.Identity()

            #
            self.encoder.append(down)

        # mid block with attention
        self.mid = nn.Sequential(
            Block(in_channels=channels),
            Attention(channels=channels),
            Block(in_channels=channels),
        )

        # output encoder
        self.encoder_output = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=channels,
                      out_channels=2 * z_channels if double_z else z_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """

        # input convolution
        x = self.encoder_input(x)

        # Top-level blocks
        for encoder in self.encoder:
            # Blocks
            for block in encoder.blocks:
                x = block(x)
            # Down-sampling
            x = encoder.downSample(x)

        # mid block with attention
        x = self.mid(x)

        # Map image space to mean-var in z space
        x = self.encoder_output(x)

        #
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    encoder = Encoder(in_channels=3)
    out = encoder(x)

    print('***** Encoder *****')
    print('Input:', x.shape)
    print('Output:', out.shape)