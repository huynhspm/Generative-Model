from typing import List

import torch
import pyrootutils
from torch import nn
from torch import Tensor

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.blocks import init_block
from src.models.components.attentions import init_attention
from src.models.components.up_down import UpSample

class Decoder(nn.Module):
    """
    ## Decoder module
    """

    def __init__(self,
                 out_channels: int,
                 z_channels: int = 3,
                 base_channels: int = 64,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4],
                 attention: str = "Attention") -> None:
        """
        out_channels: is the number of channels in the input
        z_channels: is the number of channels in the embedding space
        base_channels: is the number of channels in the final convolution layer
        block: is the block in each layers of decoder
        n_layer_blocks: is the number of resnet layers at each resolution
        channel_multipliers: are the multiplicative factors for the number of channels in the subsequent blocks
        attention: 
        """
        super().__init__()

        # Number of levels downSample
        levels = len(channel_multipliers)

        # Number of channels at each level
        channels_list = [base_channels * m for m in channel_multipliers]

        channels = base_channels

        # block to upSample
        Block = init_block(block)

         # attention layer
        Attention = init_attention(attention)

        # Number of channels in the  top-level block
        channels = channels_list[-1]

        # map z space to image space
        self.decoder_input = nn.Conv2d(in_channels=z_channels,
                                 out_channels=channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        # mid block with attention
        self.mid = nn.Sequential(
            Block(channels, channels),
            Attention(channels=channels),
            Block(channels, channels),
        )

        # List of top-level blocks
        self.decoder = nn.ModuleList()

        # prepare layer for upSampling
        for i in reversed(range(levels)):
            # Add the blocks, attentions and upSample
            blocks = nn.ModuleList()

            for _ in range(n_layer_blocks + 1):
                blocks.append(
                    Block(
                        in_channels=channels,
                        out_channels=channels_list[i],
                    ))

                channels = channels_list[i]
                
            up = nn.Module()
            up.blocks = blocks

            # Up-sampling at the end of each top level block except the first
            if i != 0:
                up.upSample = UpSample(channels=channels)
            else:
                up.upSample = nn.Identity()

            # Prepend to be consistent with the checkpoint
            self.decoder.insert(0, up)

        # output convolution
        self.decoder_output = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        :param z: is the embedding tensor with shape `[batch_size, z_channels, z_height, z_height]`
        """

        # Map z space to image space
        x = self.decoder_input(z)
        
        # mid block with attention
        x = self.mid(x)

        # Top-level blocks
        for decoder in reversed(self.decoder):
            # Blocks
            for block in decoder.blocks:
                x = block(x)
            # Up-sampling
            x = decoder.upSample(x)

        # output convolution
        x = self.decoder_output(x)

        #
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 32, 8, 8)
    decoder = Decoder(out_channels=3)
    out = decoder(x)

    print('***** Decoder *****')
    print('Input:', x.shape)
    print('Output:', out.shape)