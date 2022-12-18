import torch
import torch.nn as nn
from typing import List
from diffusion_model.denoise_models.models.modules import TimestepEmbedSequential, UpSample, DownSample
from diffusion_model.denoise_models.attentions.self_attention_wrapper import SAWrapper
from diffusion_model.denoise_models.attentions.spatial_transformer import SpatialTransformer
from diffusion_model.denoise_models.backbones.base import BaseBlock
from diffusion_model.denoise_models.backbones.vgg import VGGBlock
from diffusion_model.denoise_models.backbones.resnet import ResnetBlock

Attentions = {
    "SAWrapper": SAWrapper,
    "Transformer": SpatialTransformer,
}

Backbones = {
    "Base": BaseBlock,
    "VGG": VGGBlock,
    "Resnet": ResnetBlock,
}


class UnetModel(nn.Module):
    '''
        in_channels: the number of channels in the input feature map
        out_channels: is the number of channels in the output feature map
        size: width = height of image
        channels: the base channel count for the model
        attention: name of attentions
        block: name of blocks for each level
        n_layer_blocks: number of blocks at each level
        attention_levels: the levels at which attention should be performed
        channel_multipliers:  the multiplicative factors for number of channels for each level
        tf_heads: is the number of attention heads
        tf_layers: is the number of transformer layers
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 size: int,
                 channels: int,
                 attention_name: str,
                 backbone: str,
                 n_layer_blocks: int,
                 attention_levels: List[int],
                 channel_multipliers: List[int],
                 tf_heads: int = 4,
                 tf_layers: int = 1) -> None:
        super().__init__()

        # Size time embeddings
        d_time_emb = channels * channel_multipliers[-1]

        # layer for time embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()

        # convolution for original image
        self.input_blocks.append(
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, channels, 3, padding=1)))

        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]

        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # Number of levels (downSample and upSample)
        levels = len(channel_multipliers)

        # attention layer
        Attention = Attentions[attention_name]

        #block for each level
        Block = Backbones[backbone]

        # Prepare for input half of U-net
        for i in range(levels):
            # Add the blocks and attentions
            for _ in range(n_layer_blocks):
                layers = [
                    Block(in_channels=channels,
                          d_t_emb=d_time_emb,
                          out_channels=channels_list[i])
                ]

                channels = channels_list[i]
                input_block_channels.append(channels)

                # add attention layer
                if i in attention_levels:
                    layers.append(Attention(channels, size))

                self.input_blocks.append(TimestepEmbedSequential(*layers))

            # Down sample at all levels except last
            if i != levels - 1:
                size //= 2
                self.input_blocks.append(
                    TimestepEmbedSequential(DownSample(channels=channels)))
                input_block_channels.append(channels)

    # The middle of the U-Net
        self.middle_block = TimestepEmbedSequential(
            Block(in_channels=channels, d_t_emb=d_time_emb),
            Attention(channels, size),
            Block(in_channels=channels, d_t_emb=d_time_emb),
        )

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])

        for i in reversed(range(levels)):
            for j in range(n_layer_blocks + 1):
                layers = [
                    Block(in_channels=channels + input_block_channels.pop(),
                          d_t_emb=d_time_emb,
                          out_channels=channels_list[i])
                ]
                channels = channels_list[i]

                # add attention layer
                if i in attention_levels:
                    layers.append(Attention(channels, size))

                if i != 0 and j == n_layer_blocks:
                    layers.append(UpSample(channels))

                self.output_blocks.append(TimestepEmbedSequential(*layers))

            size *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
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
        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)

        # Middle of the U-Net
        x = self.middle_block(x, t_emb, cond)

        # Output half of the U-Net
        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        # Final normalization and $3 \times 3$ convolution
        return self.out(x)