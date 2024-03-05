from typing import List

import torch
import pyrootutils
from torch import nn
from torch import Tensor

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.blocks import init_block
from src.models.components.up_down import DownSample


class ImageEmbedder(nn.Module):
    """
    ### Image Embedder module
    """

    def __init__(self,
                 in_channels: int,
                 channels: int = 32,
                 d_embed: int = 3,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4]) -> None:
        """
        in_channels: is the number of channels in the image
        channels: is the number of channels in the first convolution layer
        d_embed: is the number of channels in the embedding space
        block: is the block of block in each layers of embedder
        n_layer_blocks: is the number of resnet layers at each resolution
        channel_multipliers: are the multiplicative factors for the number of channels in the subsequent blocks
        """
        super().__init__()

        # Number of levels downSample
        levels = len(channel_multipliers)

        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # Block to downSample
        Block = init_block(block) if block is not None else None

        # Input convolution
        self.embedder_input = nn.Conv2d(in_channels=in_channels,
                                        out_channels=channels,
                                        kernel_size=3,
                                        padding=1)

        # List of top-level blocks
        self.embedder = nn.ModuleList()

        # Prepare layer for downSampling
        for i in range(levels):
            # Add the blocks, attentions and downSample
            blocks = nn.ModuleList()

            if Block is not None:
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
            self.embedder.append(down)

        # output embedder
        self.embedder_output = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=channels,
                      out_channels=d_embed,
                      kernel_size=3,
                      stride=1,
                      padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """

        # input convolution
        x = self.embedder_input(x)

        # Top-level blocks
        for embedder in self.embedder:
            # Blocks
            for block in embedder.blocks:
                x = block(x)
            # Down-sampling
            x = embedder.downSample(x)

        x = self.embedder_output(x)

        #
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    embedder = ImageEmbedder(in_channels=3)
    out = embedder(x)

    print('***** Embedder *****')
    print('Input:', x.shape)
    print('Output:', out.shape)