from typing import List

import torch
import pyrootutils
import torch.nn as nn
import torch.nn.functional as F

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.models.components.backbones import init_backbone
from src.models.components.attentions import init_attention


class Decoder(nn.Module):
    """
    ## Decoder module
    """

    def __init__(self,
                 out_channels: int,
                 channels: int = 64,
                 z_channels: int = 4,
                 backbone: str = "ResNet",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4, 4],
                 attention: str = "Attention",
                 attention_levels: List[int] = [1, 2],
                 n_attention_heads: int = 4,
                 n_attention_layers: int = 1) -> None:
        """
        out_channels: is the number of channels in the image
        channels: is the number of channels in the final convolution layer
        z_channels: is the number of channels in the embedding space
        backbone: is the block of backbone in each layers of decoder
        n_layer_blocks: is the number of resnet layers at each resolution
        channel_multipliers: are the multiplicative factors for the number of channels in the subsequent blocks
        attention: 
        attention_levels: the levels at which attention should be performed
        n_attention_heads: the number of attention heads
        n_attention_layers: the number of attention layers
        """
        super().__init__()

        # Number of levels downSample
        levels = len(channel_multipliers)

        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # backbone to upSample
        Block = init_backbone(backbone)

         # attention layer
        Attention = init_attention(attention)

        # Number of channels in the  top-level block
        channels = channels_list[-1]

        # map latent space to image space
        self.conv_in = nn.Conv2d(in_channels=z_channels,
                                 out_channels=channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        # mid block with attention
        self.mid = nn.Sequential(
            Block(channels, channels),
            Attention(
                channels=channels,
                n_heads=n_attention_heads,
                n_layers=n_attention_layers,
            ),
            Block(channels, channels),
        )

        # List of top-level blocks
        self.up = nn.ModuleList()

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

                # add attention layer
                if i in attention_levels:
                    blocks.append(
                        Attention(
                            channels=channels,
                            n_heads=n_attention_heads,
                            n_layers=n_attention_layers,
                        ))

            up = nn.Module()
            up.blocks = blocks

            # Up-sampling at the end of each top level block except the first
            if i != 0:
                up.upSample = UpSample(channels=channels)
            else:
                up.upSample = nn.Identity()

            # Prepend to be consistent with the checkpoint
            self.up.insert(0, up)

        # output convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        :param z: is the embedding tensor with shape `[batch_size, z_channels, z_height, z_height]`
        """

        # Map latent space to image space
        x = self.conv_in(z)

        # mid block with attention
        x = self.mid(x)

        # Top-level blocks
        for up in reversed(self.up):
            # Blocks
            for block in up.blocks:
                x = block(x)
            # Up-sampling
            x = up.upSample(x)

        # output convolution
        x = self.conv_out(x)

        #
        return x


class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int) -> None:
        """
        channels: number of channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=3,
                              padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: input feature map: [batch_size, channels, height, width]

        """
        # Up-sample by a factor of 2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "net")
    
    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="decoder.yaml")
    def main(cfg: DictConfig):
        decoder: Decoder = hydra.utils.instantiate(cfg.get('decoder'))
        x = torch.randn(2, 4, 4, 4)
        print('input:', x.shape)       
        out = decoder(x)
        print('output decoder:', out.shape)
    
    main()