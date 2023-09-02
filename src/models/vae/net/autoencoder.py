from typing import Any, List, Tuple

import torch
import pyrootutils
from torch import nn
from torch import Tensor
import torch.nn.functional as F

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vae.net import BaseVAE
from src.models.vae.net.base import GaussianDistribution
from src.models.components.blocks import init_block
from src.models.components.attentions import init_attention
from src.models.components.sample import DownSample, UpSample


class AutoEncoder(BaseVAE):
    """
    ## AutoEncoder

    This consists of the encoder and decoder modules.
    """

    def __init__(
        self,
        img_dims: int = [3, 32, 32],
        z_channels: int = 32,
        channels: int = 32,
        block: str = 'Residual',
        n_layer_blocks: int = 1,
        channel_multipliers: List[int] = [1, 2, 4],
        attention: str = 'Attention',
        kld_weight: Tuple[int, int] = [0, 1]) -> None:
        """
        encoder:
        decoder:
        """
        super(AutoEncoder, self).__init__()
        self.kld_weight = kld_weight[0] / kld_weight[1]

        self.encoder = Encoder(in_channels=img_dims[0],
                               channels=channels,
                               z_channels=z_channels,
                               block=block,
                               n_layer_blocks=n_layer_blocks,
                               channel_multipliers=channel_multipliers,
                               attention=attention)
        
        self.decoder = Decoder(out_channels=img_dims[0],
                               channels=channels,
                               z_channels=z_channels,
                               block=block,
                               n_layer_blocks=n_layer_blocks,
                               channel_multipliers=channel_multipliers,
                               attention=attention)

        self.latent_dims = [z_channels,
                            int(img_dims[1] / (len(channel_multipliers) - 1)), 
                            int(img_dims[2] / (len(channel_multipliers) - 1))]

    def encode(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        """
        ### Encode images to z representation

        img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_width]`
        mean_var = self.encoder(img)
        z, kld_loss = GaussianDistribution(mean_var).sample()
        return z, kld_loss

    def decode(self, z: Tensor) -> Tensor:
        """
        ### Decode images from latent s

        z: is the z representation with shape `[batch_size, z_channels, z_height, z_width]`
        """

        # Decode the image of shape `[batch_size, img_channels, img_height, img_width]`
        return self.decoder(z)

    def forward(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        z, kld_loss = self.encode(img)
        return self.decode(z), kld_loss
    
    def loss_function(self, 
                      img: Tensor, 
                      recons_img: Tensor, 
                      kld_loss: float) -> Tensor:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """

        recons_loss = F.mse_loss(recons_img, img)
        loss = recons_loss + self.kld_weight * kld_loss
        return {'loss': loss, 
                'Reconstruction_Loss':recons_loss.detach(), 
                'KLD':kld_loss.detach()}
    
class Encoder(nn.Module):
    """
    ### Encoder module
    """

    def __init__(self,
                 in_channels: int,
                 channels: int = 32,
                 z_channels: int = 32,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4],
                 attention: str = "Attention") -> None:
        """
        in_channels: is the number of channels in the image
        channels: is the number of channels in the first convolution layer
        z_channels: is the number of channels in the embedding space
        block: is the block of block in each layers of encoder
        n_layer_blocks: is the number of resnet layers at each resolution
        channel_multipliers: are the multiplicative factors for the number of channels in the subsequent blocks
        """
        super().__init__()

        # Number of levels downSample
        levels = len(channel_multipliers)

        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

        # Block to downSample
        Block = init_block(block)

        # attention layer
        Attention = init_attention(attention)

        # Number of channels in the  top-level block
        channels = channels_list[-1]

        # Input convolution
        self.encoder_input = nn.Conv2d(in_channels=in_channels,
                                 out_channels=channels,
                                 kernel_size=3,
                                 padding=1)

        # List of top-level blocks
        self.encoder = nn.ModuleList()

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
                Block(in_channels=channels),
                Attention(channels=channels),
                Block(in_channels=channels),
            )

        # Map to embedding space: 2*z_channels to divide into mean and log_var
        self.encoder_output = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=channels,
                        out_channels=2*z_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """

        # input convolution
        x = self.encoder_input(x)

        # Top-level blocks
        for encoder in self.encoder:
            # Blocks
            for block in encoder.blocks:
                x = block(x)
            # Down-sampling
            x = encoder.downSample(x)

        # mid block with attention
        x = self.mid(x)

        # Map image space to mean-var in z space
        mean_var = self.encoder_output(x)

        #
        return mean_var
    
class Decoder(nn.Module):
    """
    ## Decoder module
    """

    def __init__(self,
                 out_channels: int,
                 channels: int = 32,
                 z_channels: int = 32,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4],
                 attention: str = "Attention") -> None:
        """
        out_channels: is the number of channels in the image
        channels: is the number of channels in the final convolution layer
        z_channels: is the number of channels in the embedding space
        block: is the block in each layers of decoder
        n_layer_blocks: is the number of resnet layers at each resolution
        channel_multipliers: are the multiplicative factors for the number of channels in the subsequent blocks
        attention: 
        """
        super().__init__()

        # Number of levels downSample
        levels = len(channel_multipliers)

        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]

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
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "vae" / "net")
    
    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="autoencoder.yaml")
    def main(cfg: DictConfig):
        cfg.kld_weight = [0, 1]

        # print(cfg)
        autoencoder: AutoEncoder = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 3, 32, 32)

        z, kld_loss = autoencoder.encode(x)
        out, kld_loss = autoencoder(x)
        sample = autoencoder.sample(n_samples=2)

        print('***** AutoEncoder *****')
        print('Input:', x.shape)
        print('Encode:', z.shape)
        print('KLD_Loss:', kld_loss.detach())
        print('Output:', out.shape)
        print('Sample:', sample.shape)

    main()