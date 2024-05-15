from typing import List, Dict

import torch
import pyrootutils
import torch.nn as nn
from torch import Tensor

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.blocks import init_block, get_all_blocks
from src.models.components.attentions import init_attention, get_all_attentions
from src.models.components.embeds import TimeEmbedding
from src.models.components.up_down import DownSample, UpSample


class UNetAttention(nn.Module):
    """
    ### UNet model with attention
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 base_channels: int = 64,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4],
                 attention: str = "SelfAttention",
                 attention_levels: List[int] = [1, 2],
                 n_attention_heads: int = 4,
                 n_attention_layers: int = 1,
                 d_cond_image: int = None,
                 d_cond_text: int = None,
                 drop_rate: float = 0.) -> None:
        """_summary_

        Args:
            in_channels (int): the number of channels in the input
            out_channels (int): the number of channels in the output
            base_channels (int, optional): the base channel count for the model. Defaults to 64.
            block (str, optional): type of block for each level. Defaults to "Residual".
            n_layer_blocks (int, optional): number of blocks at each level. Defaults to 1.
            channel_multipliers (List[int], optional): the multiplicative factors for number of channels for each level. Defaults to [1, 2, 4].
            attention (str, optional): type of attentions for each level. Defaults to "SelfAttention".
            attention_levels (List[int], optional): the levels at which attention should be performed. Defaults to [1, 2].
            n_attention_heads (int, optional): the number of attention heads. Defaults to 4.
            n_attention_layers (int, optional): the number of attention layers. Defaults to 1.
            d_cond_image (int, optional): the number of dimension of image condition. Defaults to None.
            d_cond_text (int, optional): the number of dimension of text condition. Defaults to None.
            drop_rate (float, optional): percentage of dropout. Defaults to 0..
        """
        super().__init__()

        # size time embeddings
        d_time_emb = base_channels * channel_multipliers[-1]

        # layer for time embeddings
        self.time_embed = TimeEmbedding(base_channels, d_time_emb)

        # number of levels (downSample and upSample)
        levels = len(channel_multipliers)

        # number of channels at each level
        channels_list = [base_channels * m for m in channel_multipliers]

        channels = base_channels

        # block to downSample
        Block = init_block(block)

        # attention layer
        Attention = init_attention(
            attention) if attention is not None else None

        # input half of the U-Net
        self.down = nn.ModuleList()

        # input convolution
        self.down.append(
            SequentialBlock(
                nn.Conv2d(in_channels=in_channels +
                          (d_cond_image if d_cond_image is not None else 0),
                          out_channels=channels,
                          kernel_size=3,
                          padding=1)))

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
                            d_cond=d_cond_text,
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
                d_cond=d_cond_text,
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
                            d_cond=d_cond_text,
                        ))

                if i != 0 and j == n_layer_blocks:
                    layers.append(UpSample(channels))

                self.up.append(SequentialBlock(*layers))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
        )

    def forward(self,
                x: Tensor,
                time_steps: Tensor,
                cond: Dict[str, Tensor] = None) -> Tensor:
        """_summary_

        
        Args:
            x (Tensor): is the input feature map of shape `[batch_size, channels, width, height]`
            time_steps (Tensor): are the time steps of shape `[batch_size]`
            cond (Dict[str, Tensor], optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """

        # get time step embeddings
        t_emb = self.time_embed(time_steps)
        text_embed = None

        if cond is not None:
            if 'label' in cond.keys():
                assert cond['label'].shape[0] == x.shape[0], 'shape not match'
                t_emb = t_emb + cond['label']

            if 'image' in cond.keys():
                assert cond['image'].shape[0] == x.shape[0], 'shape not match'
                x = torch.cat((x, cond['image']), dim=1)

            if 'text' in cond.keys():
                assert cond['text'].shape[0] == x.shape[0], 'shape not match'
                text_embed = cond['text']

        # to store the input half outputs for skip connections
        x_input_block = []

        # input half of the U-Net
        for module in self.down:
            x = module(x, t_emb, text_embed)
            x_input_block.append(x)

        # middle of the U-Net
        x = self.mid(x, t_emb, text_embed)

        # Output half of the U-Net
        for module in self.up:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, text_embed)

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
    config_path = str(root / "configs" / "model" / "unet" / "net")
    print("root: ", root)

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="unet_attention.yaml")
    def main(cfg: DictConfig):
        # print(cfg)

        unet_attention: UNetAttention = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 1000, (2, ))
        out1 = unet_attention(x, t)
        print('***** UNet Attention *****')
        print('Input:', x.shape)
        print('Output:', out1.shape)

        print('-' * 60)

        cfg['d_cond_image'] = 1
        # print(cfg)

        cond_unet_attention: UNetAttention = hydra.utils.instantiate(cfg)
        cond = {
            'label': torch.randn(2, 256),
            'image': torch.rand_like(x),
        }

        out2 = cond_unet_attention(x, t, cond=cond)
        print('***** Condition UNet Attention *****')
        print('Input:', x.shape)
        print('Cond_label:', cond['label'].shape)
        print('Cond_image:', cond['image'].shape)
        print('Output:', out2.shape)

    main()
