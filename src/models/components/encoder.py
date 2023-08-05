from typing import List

import torch
import pyrootutils
import torch.nn as nn

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.models.components.backbones import init_backbone
from src.models.components.attentions import init_attention


class Encoder(nn.Module):
    """
    ### Encoder module
    """

    def __init__(self,
                 in_channels: int,
                 channels: int = 64,
                 z_channels: int = 4,
                 backbone: str = "Base",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4, 4],
                 attention: str = "Attention",
                 attention_levels: List[int] = [1, 2],
                 n_attention_heads: int = 4,
                 n_attention_layers: int = 1) -> None:
        """
        in_channels: is the number of channels in the image
        channels: is the number of channels in the first convolution layer
        z_channels: is the number of channels in the embedding space
        backbone: is the block of backbone in each layers of encoder
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

        # Backbone to downSample
        Block = init_backbone(backbone)

        # attention layer
        Attention = init_attention(attention)

        # Number of channels in the  top-level block
        channels = channels_list[-1]

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels=in_channels,
                                 out_channels=channels,
                                 kernel_size=3,
                                 padding=1)

        # List of top-level blocks
        self.down = nn.ModuleList()

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

                # add attention layer
                if i in attention_levels:
                    blocks.append(
                        Attention(
                            channels=channels,
                            n_heads=n_attention_heads,
                            n_layers=n_attention_layers,
                        ))

            down = nn.Module()
            down.blocks = blocks

            # Down-sampling at the end of each top level block except the last
            if i != levels - 1:
                down.downSample = DownSample(channels=channels)
            else:
                down.downSample = nn.Identity()

            #
            self.down.append(down)

            # mid block with attention
            self.mid = nn.Sequential(
                Block(in_channels=channels),
                Attention(
                    channels=channels,
                    n_heads=n_attention_heads,
                    n_layers=n_attention_layers,
                ),
                Block(in_channels=channels),
            )

            # Map to embedding space
            self.conv_out = nn.Sequential(
                nn.GroupNorm(num_groups=32, num_channels=channels),
                nn.SiLU(),
                nn.Conv2d(in_channels=channels,
                          out_channels=2 * z_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """

        # input convolution
        x = self.conv_in(x)

        # Top-level blocks
        for down in self.down:
            # Blocks
            for block in down.blocks:
                x = block(x)
            # Down-sampling
            x = down.downSample(x)

        # mid block with attention
        x = self.mid(x)

        # Map image space to latent space
        z = self.conv_out(x)

        #
        return z


class DownSample(nn.Module):
    """
    ### Down-sampling layer
    """

    def __init__(self, channels: int) -> None:
        """
        channels: is the number of channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels,
                              out_channels=channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
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
                config_name="encoder.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        encoder: Encoder = hydra.utils.instantiate(cfg.get('encoder'))
        x = torch.randn(2, 1, 32, 32)
        print('input:', x.shape)
        out = encoder(x)
        print('output encoder:', out.shape)
   
    main()