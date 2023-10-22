from typing import List

import torch
import pyrootutils
import torch.nn as nn
from torch import Tensor

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.blocks import init_block, get_all_blocks
from src.models.components.attentions import init_attention, get_all_attentions
from src.models.components.embeds import TimeEmbedding
from src.models.components.up_down import DownSample, UpSample


class UNet(nn.Module):
    """
    ### Unet model
    """

    def __init__(self,
                 img_channels: int,
                 channels: int = 64,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4, 4],
                 attention: str = "SelfAttention",
                 attention_levels: List[int] = [1, 2],
                 n_attention_heads: int = 4,
                 n_attention_layers: int = 1,
                 d_cond: int = None,
                 drop_rate: float = 0.) -> None:
        """
        img_channels: the number of channels in the input feature map
        channels: the base channel count for the model
        block: name of block for each level
        n_layer_blocks: number of blocks at each level
        channel_multipliers: the multiplicative factors for number of channels for each level
        attention: name of attentions for each level
        attention_levels: the levels at which attention should be performed
        n_attention_heads: the number of attention heads
        n_attention_layers: the number of attention layers
        d_cond: the number of dimension of condition
        drop_rate: percentage of dropout
        """
        super().__init__()

        self.base_channels = channels

        # size time embeddings
        d_time_emb = channels * channel_multipliers[-1]

        # layer for time embeddings
        self.time_embed = TimeEmbedding(channels, d_time_emb)

        # number of levels (downSample and upSample)
        levels = len(channel_multipliers)

        # number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # block to downSample
        Block = init_block(block)

        # attention layer
        Attention = init_attention(
            attention) if attention is not None else None

        # input half of the U-Net
        self.down = nn.ModuleList()

        # input convolution
        self.down.append(
            SequentialBlock(nn.Conv2d(img_channels, channels, 3, padding=1)))

        # number of channels at each block in the input half of U-Net
        input_block_channels = [channels]

        # prepare for input half of U-net
        for i in range(levels):
            # add the blocks, attentions
            for _ in range(n_layer_blocks):
                layers = [
                    Block(
                        in_channels=channels,
                        d_t_emb=d_time_emb,
                        out_channels=channels_list[i],
                        drop_rate=drop_rate,
                    )
                ]

                channels = channels_list[i]
                input_block_channels.append(channels)

                # add attention layer
                if attention is not None and i in attention_levels:
                    layers.append(
                        Attention(
                            channels=channels,
                            n_heads=n_attention_heads,
                            n_layers=n_attention_layers,
                            d_cond=d_cond,
                        ))

                self.down.append(SequentialBlock(*layers))

            # down sample at all levels except last
            if i != levels - 1:
                self.down.append(SequentialBlock(
                    DownSample(channels=channels)))
                input_block_channels.append(channels)

        # the middle of the U-Net
        self.mid = SequentialBlock(
            Block(
                in_channels=channels,
                d_t_emb=d_time_emb,
                drop_rate=drop_rate,
            ),
            Attention(
                channels=channels,
                n_heads=n_attention_heads,
                n_layers=n_attention_layers,
                d_cond=d_cond,
            ) if attention is not None else Block(
                in_channels=channels,
                d_t_emb=d_time_emb,
                drop_rate=drop_rate,
            ),
            Block(
                in_channels=channels,
                d_t_emb=d_time_emb,
                drop_rate=drop_rate,
            ),
        )

        # second half of the U-Net
        self.up = nn.ModuleList([])

        # prepare layer for upSampling
        for i in reversed(range(levels)):
            # add the blocks, attentions

            for j in range(n_layer_blocks + 1):
                layers = [
                    Block(
                        in_channels=channels + input_block_channels.pop(),
                        d_t_emb=d_time_emb,
                        out_channels=channels_list[i],
                        drop_rate=drop_rate,
                    )
                ]
                channels = channels_list[i]

                # add attention layer
                if attention is not None and i in attention_levels:
                    layers.append(
                        Attention(
                            channels=channels,
                            n_heads=n_attention_heads,
                            n_layers=n_attention_layers,
                            d_cond=d_cond,
                        ))

                if i != 0 and j == n_layer_blocks:
                    layers.append(UpSample(channels))

                self.up.append(SequentialBlock(*layers))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, img_channels, 3, padding=1),
        )

    def forward(self,
                x: Tensor,
                time_steps: Tensor,
                cond: Tensor | None = None):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, 1]`
        """

        # to store the input half outputs for skip connections
        x_input_block = []

        # get time step embeddings
        t_emb = self.time_embed(time_steps)

        # input half of the U-Net
        for module in self.down:
            x = module(x, t_emb, cond)
            x_input_block.append(x)

        # middle of the U-Net
        x = self.mid(x, t_emb, cond)

        # Output half of the U-Net
        for module in self.up:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        # output convolution
        x = self.conv_out(x)

        #
        return x


class SequentialBlock(nn.Sequential):
    """
    ### Sequential block for modules with different inputs
    This sequential module can compose of different modules suck as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self, x: Tensor, t_emb: Tensor = None, cond: Tensor = None):
        for layer in self:
            if isinstance(layer, get_all_blocks()):
                x = layer(x, t_emb)
            elif isinstance(layer, get_all_attentions()):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    config_path = str(root / "configs" / "model" / "unet")
    print("root: ", root)

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="unet.yaml")
    def main(cfg: DictConfig):
        # print(cfg)

        cfg = cfg['denoise_net']
        unet: UNet = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 1000, (2, ))
        out1 = unet(x, t)
        print('***** UNet *****')
        print('Input:', x.shape)
        print('Output:', out1.shape)

        print('-' * 60)

        cfg.n_classes = 2
        cfg.attention = 'CrossAttention'
        cond = torch.randint(0, 2, (2, ))
        cond_unet: UNet = hydra.utils.instantiate(cfg)
        out2 = cond_unet(x, t, cond=cond)
        print('***** Condition_UNet *****')
        print('Input:', x.shape)
        print('Output:', out2.shape)

    main()
