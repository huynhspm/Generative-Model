from typing import List, Optional

import math
import torch
import pyrootutils
import torch.nn as nn
import torch.nn.functional as F

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.backbones import init_backbone, get_all_backbones
from src.models.components.attentions import init_attention, get_all_attentions


class UNetModel(nn.Module):
    """
    ### Unet model
    """

    def __init__(self,
                 img_channels: int,
                 channels: int = 64,
                 backbone: str = "ResNet",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4, 4],
                 attention: str = "SelfAttention",
                 attention_levels: List[int] = [1, 2],
                 n_attention_heads: int = 4,
                 n_attention_layers: int = 1,
                 n_classes: int = None,
                 drop_rate: float = 0.) -> None:
        """
        img_channels: the number of channels in the input feature map
        channels: the base channel count for the model
        backbone: name of block backbone for each level
        n_layer_blocks: number of blocks at each level
        channel_multipliers: the multiplicative factors for number of channels for each level
        attention: name of attentions for each level
        attention_levels: the levels at which attention should be performed
        n_attention_heads: the number of attention heads
        n_attention_layers: the number of attention layers
        n_classes: the number of classes
        """
        super().__init__()

        self.base_channels = channels

        # size time embeddings
        d_time_emb = channels * channel_multipliers[-1]

        # layer for time embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

        d_cond = None
        # Size condition embeddings
        if n_classes is not None:
            d_cond = channels * channel_multipliers[-1]

            # layer for time embeddings
            self.cond_embed = nn.Sequential(
                nn.Embedding(n_classes, d_cond),
                nn.SiLU(),
                nn.Linear(d_cond, d_cond),
            )

        # number of levels (downSample and upSample)
        levels = len(channel_multipliers)

        # number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # backbone to downSample
        Block = init_backbone(backbone)

        # attention layer
        Attention = init_attention(attention)

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
                if i in attention_levels:
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
                if i in attention_levels:
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
            nn.SiLU(),
            nn.Conv2d(channels, img_channels, 3, padding=1),
        )

    def forward(self,
                x: torch.Tensor,
                time_steps: torch.Tensor,
                cond: Optional[torch.Tensor] = None):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, 1]`
        """

        # to store the input half outputs for skip connections
        x_input_block = []

        # get time embeddings: sin, cos embedding
        t_emb = self.time_step_embedding(time_steps)

        # get time step embeddings
        t_emb = self.time_embed(t_emb)

        if cond is not None:
            cond = self.cond_embed(cond)

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

    def time_step_embedding(self,
                            time_steps: torch.Tensor,
                            max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings
        time_steps: are the time steps of shape `[batch_size]`
        max_period: controls the minimum frequency of the embeddings.
        """

        half = self.base_channels // 2
        frequencies = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) /
            half).to(device=time_steps.device)

        args = time_steps[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


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
        # up-sample by a factor of 2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # apply convolution
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
        # apply convolution
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
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    config_path = str(root / "configs" / "model" / "net" / "denoise_model")
    print("root: ", root)

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="unet.yaml")
    def main(cfg: DictConfig):
        unet_model: UNetModel = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 1, 32, 32)
        t = torch.randn(2)
        cond = torch.randint(0, 2, (2, ))
        out = unet_model(x, t, cond)
        print(out.shape)

        # test sin-cos time embedding
        test_time_embeddings(unet_model)

    def test_time_embeddings(unet_model: UNetModel):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        timesteps = torch.arange(0, 1000)
        te = unet_model.time_step_embedding(timesteps)
        plt.plot(timesteps, te[:, [10, 20, 40, 60]])
        plt.legend(["dim %d" % p for p in [10, 20, 40, 60]])
        plt.title("Time embeddings")
        plt.show()

    main()
