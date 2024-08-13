from typing import List

import torch
import pyrootutils
from torch import nn
from torch import Tensor

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.blocks import init_block
from src.models.components.attentions import init_attention
from src.models.components.up_down import DownSample

# Follow Stable Diffusion: https://nn.labml.ai/diffusion/stable_diffusion/model/autoencoder.html

class Encoder(nn.Module):
    """
    ### Encoder module
    """

    def __init__(self,
                 in_channels: int,
                 z_channels: int = 3,
                 base_channels: int = 64,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 drop_rate: float = 0.,
                 channel_multipliers: List[int] = [1, 2, 4],
                 attention: str = "Attention",
                 n_attention_heads: int | None = None,
                 n_attention_layers: int | None = None,
                 double_z: bool = False) -> None:
        """_summary_

        Args:
            in_channels (int): is the number of channels in the input.
            z_channels (int, optional): is the number of channels in the embedding space. Defaults to 3.
            base_channels (int, optional): is the number of channels in the first convolution layer. Defaults to 64.
            block (str, optional): is the block of block in each layers of encoder. Defaults to "Residual".
            n_layer_blocks (int, optional): is the number of resnet layers at each resolution. Defaults to 1.
            drop_rate (float, optional): parameter of dropout layer. Defaults to 0..
            channel_multipliers (List[int], optional): the multiplicative factors for number of channels for each level. Defaults to [1, 2, 4].
            attention (str, optional): type of attentions for each level. Defaults to "Attention".
            n_attention_heads (int, optional): the number of head for multi-head attention. Defaults to None.
            n_attention_layers (int, optional): the number of layer in each attention. Defaults to None.
            double_z (bool, optional): _description_. Defaults to False.
        """

        super().__init__()

        # Number of levels downSample
        levels = len(channel_multipliers)

        # Number of channels at each level
        channels_list = [base_channels * m for m in channel_multipliers]

        channels = base_channels

        # Block to downSample
        Block = init_block(block)

        # attention layer
        Attention = init_attention(
            attention) if attention is not None else None

        # Input convolution
        self.encoder_input = nn.Conv2d(in_channels=in_channels,
                                       out_channels=channels,
                                       kernel_size=3,
                                       padding=1)

        # List of top-level blocks
        self.encoder = nn.ModuleList()

        # Prepare layer for downSampling
        for i in range(levels):
            # Add the blocks and downSample
            blocks = nn.ModuleList()

            for _ in range(n_layer_blocks):
                blocks.append(
                    Block(in_channels=channels,
                          out_channels=channels_list[i],
                          drop_rate=drop_rate))

                channels = channels_list[i]

            down = nn.Module()
            down.blocks = blocks

            # Down-sampling at the end of each top level block except the last
            if i != levels - 1:
                down.downSample = DownSample(channels=channels)
            else:
                down.downSample = nn.Identity()

            #
            self.encoder.append(down)

        # mid block with attention
        self.mid = nn.Sequential(
            Block(in_channels=channels, drop_rate=drop_rate),
            Attention(channels=channels, 
                      n_heads=n_attention_heads,
                      n_layers=n_attention_layers) if attention is not None \
                      else Block(in_channels=channels, drop_rate=drop_rate),
            Block(in_channels=channels, drop_rate=drop_rate))

        # output encoder
        self.encoder_output = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=channels,
                      out_channels=2 * z_channels if double_z else z_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1))

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`

        Returns:
            Tensor: _description_
        """
        
        # input convolution
        z = self.encoder_input(x)

        # Top-level blocks
        for encoder in self.encoder:
            # Blocks
            for block in encoder.blocks:
                z = block(z)
            # Down-sampling
            z = encoder.downSample(z)

        # mid block with attention
        z = self.mid(z)

        # Map image space to mean-var in z space
        z = self.encoder_output(z)

        return z


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    config_path = str(root / "configs" / "model" / "components" / "up_down")
    print("root: ", root)

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="encoder.yaml")
    def main(cfg: DictConfig):
        print(cfg)

        encoder: Encoder = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 3, 32, 32)

        z = encoder(x)

        print('***** Encoder *****')
        print('Input:', x.shape)
        print('Output:', z.shape)

    main()