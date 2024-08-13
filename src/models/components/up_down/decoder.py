from typing import List

import torch
import pyrootutils
from torch import nn
from torch import Tensor

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.blocks import init_block
from src.models.components.attentions import init_attention
from src.models.components.up_down import UpSample

# Follow Stable Diffusion: https://nn.labml.ai/diffusion/stable_diffusion/model/autoencoder.html
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
                drop_rate: float = 0.,
                channel_multipliers: List[int] = [1, 2, 4],
                attention: str = "Attention",
                n_attention_heads: int | None = None,
                n_attention_layers: int | None = None) -> None:
        """_summary_

        Args:
            out_channels (int): is the number of channels in the output.
            z_channels (int, optional): is the number of channels in the embedding space. Defaults to 3.
            base_channels (int, optional): is the number of channels in the first convolution layer. Defaults to 64.
            block (str, optional): _description_. Defaults to "Residual".
            n_layer_blocks (int, optional): _description_. Defaults to 1.
            drop_rate (float, optional): parameter of dropout layer. Defaults to 0..
            channel_multipliers (List[int], optional): _description_. Defaults to [1, 2, 4].
            attention (str, optional): _description_. Defaults to "Attention".
            n_attention_heads (int | None, optional): _description_. Defaults to None.
            n_attention_layers (int | None, optional): _description_. Defaults to None.
        """
        super().__init__()

        # Number of levels downSample
        levels = len(channel_multipliers)

        # Number of channels at each level
        channels_list = [base_channels * m for m in channel_multipliers]

        # Number of channels in the  top-level block
        channels = channels_list[-1]

        # block to upSample
        Block = init_block(block)

        # attention layer
        Attention = init_attention(
            attention) if attention is not None else None

        # map z space to image space
        self.decoder_input = nn.Conv2d(in_channels=z_channels,
                                    out_channels=channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        # mid block with attention
        self.mid = nn.Sequential(
            Block(channels, channels),
            Attention(channels=channels, 
                    n_heads=n_attention_heads,
                    n_layers=n_attention_layers) if attention is not None \
                    else Block(in_channels=channels, drop_rate=drop_rate),
            Block(channels, channels))

        # List of top-level blocks
        self.decoder = nn.ModuleList()

        # prepare layer for upSampling
        for i in reversed(range(levels)):
            # Add the blocks and upSample
            blocks = nn.ModuleList()

            for _ in range(n_layer_blocks + 1):
                blocks.append(
                    Block(in_channels=channels,
                          out_channels=channels_list[i],
                          drop_rate=drop_rate))

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
                      padding=1))

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

        return x


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    config_path = str(root / "configs" / "model" / "components" / "up_down")
    print("root: ", root)

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="decoder.yaml")
    def main(cfg: DictConfig):
        print(cfg)

        decoder: Decoder = hydra.utils.instantiate(cfg)
        z = torch.randn(2, 3, 8, 8)

        x = decoder(z)

        print('***** Decoder *****')
        print('Input:', z.shape)
        print('Output:', x.shape)

    main()