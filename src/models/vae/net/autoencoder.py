from typing import List, Tuple

import torch
import pyrootutils
from torch import Tensor
import torch.nn.functional as F

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vae.net import BaseVAE
from src.models.vae.net.base import GaussianDistribution
from src.models.components.up_down import Encoder, Decoder


class AutoEncoder(BaseVAE):
    """
    ## AutoEncoder

    This consists of the encoder and decoder modules.
    """

    def __init__(
        self,
        img_dims: int,
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
                               attention=attention,
                               double_z=True)
        
        self.decoder = Decoder(out_channels=img_dims[0],
                               channels=channels,
                               z_channels=z_channels,
                               block=block,
                               n_layer_blocks=n_layer_blocks,
                               channel_multipliers=channel_multipliers,
                               attention=attention)

        self.latent_dims = [z_channels,
                            int(img_dims[1] / (1 << (len(channel_multipliers) - 1))), 
                            int(img_dims[2] / (1 << (len(channel_multipliers) - 1)))]

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