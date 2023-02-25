import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# import sys
# sys.path.insert(0, 'src/models')

from models.components.backbones import init_backbone, get_all_backbones
from models.components.attentions import init_attention, get_all_attentions


class UnetModel(nn.Module):
    """
    ### Unet model
    """

    def __init__(self,
                 img_channels: int,
                 channels: int = 64,
                 backbone: str = "Base",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4, 4],
                 attention: str = "Base",
                 attention_levels: List[int] = [0, 1, 2],
                 n_attention_heads: int = 4,
                 n_attention_layers: int = 1) -> None:
        """
        img_channels: the number of channels in the input feature map
        size: width = height of image
        channels: the base channel count for the model
        backbone: name of block backbone for each level
        n_layer_blocks: number of blocks at each level
        channel_multipliers: the multiplicative factors for number of channels for each level
        attention: name of attentions
        attention_levels: the levels at which attention should be performed
        n_attention_heads: the number of attention heads
        n_attention_layers: the number of attention layers
        """
        super().__init__()

        # Size time embeddings
        d_time_emb = channels * channel_multipliers[-1]

        # layer for time embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

        # Number of levels (downSample and upSample)
        levels = len(channel_multipliers)

        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # Backbone to downSample
        Block = init_backbone(backbone)

        # Attention layer
        Attention = init_attention(attention)

        # Input half of the U-Net
        self.down = nn.ModuleList()

        # Input convolution
        self.down.append(
            SequentialBlock(nn.Conv2d(img_channels, channels, 3, padding=1)))

        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]

        # Prepare for input half of U-net
        for i in range(levels):
            # Add the blocks, attentions
            for _ in range(n_layer_blocks):
                layers = [
                    Block(
                        in_channels=channels,
                        d_t_emb=d_time_emb,
                        out_channels=channels_list[i],
                    )
                ]

                channels = channels_list[i]
                input_block_channels.append(channels)

                # add attention layer
                if i in attention_levels:
                    layers.append(
                        Attention(
                            channels=channels,
                            n_heads=n_attention_heads,
                            n_layers=n_attention_layers,
                        ))

                self.down.append(SequentialBlock(*layers))

            # Down sample at all levels except last
            if i != levels - 1:
                self.down.append(SequentialBlock(
                    DownSample(channels=channels)))
                input_block_channels.append(channels)

        # The middle of the U-Net
        self.mid = SequentialBlock(
            Block(
                in_channels=channels,
                d_t_emb=d_time_emb,
            ),
            Attention(
                channels=channels,
                n_heads=n_attention_heads,
                n_layers=n_attention_layers,
            ),
            Block(
                in_channels=channels,
                d_t_emb=d_time_emb,
            ),
        )

        # Second half of the U-Net
        self.up = nn.ModuleList([])

        # prepare layer for upSampling
        for i in reversed(range(levels)):
            # Add the blocks, attentions

            for j in range(n_layer_blocks + 1):
                layers = [
                    Block(
                        in_channels=channels + input_block_channels.pop(),
                        d_t_emb=d_time_emb,
                        out_channels=channels_list[i],
                    )
                ]
                channels = channels_list[i]

                # add attention layer
                if i in attention_levels:
                    layers.append(
                        Attention(
                            channels=channels,
                            n_heads=n_attention_heads,
                            n_layers=n_attention_layers,
                        ))

                if i != 0 and j == n_layer_blocks:
                    layers.append(UpSample(channels))

                self.up.append(SequentialBlock(*layers))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, img_channels, 3, padding=1),
        )

    def forward(self,
                x: torch.Tensor,
                t_emb: torch.Tensor,
                cond: torch.Tensor = None):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param t_emb: are the time steps of shape `[batch_size, 1]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """
        # To store the input half outputs for skip connections
        x_input_block = []

        # Get time step embeddings
        t_emb = self.time_embed(t_emb)

        # Input half of the U-Net
        for module in self.down:
            x = module(x, t_emb, cond)
            x_input_block.append(x)

        # Middle of the U-Net
        x = self.mid(x, t_emb, cond)

        # Output half of the U-Net
        for module in self.up:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        # output convolution
        x = self.conv_out(x)

        #
        return x


class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of 2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)


class SequentialBlock(nn.Sequential):
    """
    ### Sequential block for modules with different inputs
    This sequential module can compose of different modules suck as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self,
                x: torch.Tensor,
                t_emb: torch.Tensor = None,
                cond: torch.Tensor = None):
        for layer in self:
            if isinstance(layer, get_all_backbones()):
                x = layer(x, t_emb)
            elif isinstance(layer, get_all_attentions()):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    t = torch.randn(1, 64)

    unet = UnetModel(img_channels=3)
    out = unet(x, t)
    print(out.shape)
