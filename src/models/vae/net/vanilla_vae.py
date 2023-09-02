from typing import List, Tuple

import torch
import pyrootutils
from torch import nn
from torch import Tensor
from torch.nn import functional as F

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vae.net import BaseVAE
from src.models.vae.net.base import GaussianDistribution


class VanillaVAE(BaseVAE):

    def __init__(self,
                 img_dims: int,
                 z_channels: int,
                 hidden_dims: List = None,
                 kld_weight: Tuple[int, int] = [0, 1]) -> None:
        super(VanillaVAE, self).__init__()

        self.kld_weight = kld_weight[0] / kld_weight[1]
        self.latent_dims = [z_channels,
                            int(img_dims[1] / len(hidden_dims)), 
                            int(img_dims[2] / len(hidden_dims))]

        in_channels = img_dims[0]
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encode_output = nn.Conv2d(in_channels=hidden_dims[-1],
                                       out_channels=2*z_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Conv2d(in_channels=z_channels,
                                 out_channels=hidden_dims[-1],
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        """
        ### Encode images to z representation

        img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_width]`
        img = self.encoder(img)
        mean_var = self.encode_output(img)
        z, kld_loss = GaussianDistribution(mean_var).sample()
        return z, kld_loss

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        img = self.decoder_input(z)
        img = self.decoder(img)
        img = self.final_layer(img)
        return img
    
    def forward(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        z, kld_loss = self.encode(img)
        return self.decode(z), kld_loss

    def loss_function(self, 
                      img: Tensor, 
                      recons_img: Tensor, 
                      kld_loss: float) -> Tensor:

        recons_loss =F.mse_loss(recons_img, img)
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
                config_name="vanilla_vae.yaml")
    def main(cfg: DictConfig):
        cfg.kld_weight = [0, 1]
        # print(cfg)

        vanillaVae: VanillaVAE = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 3, 32, 32)
        z, kld_loss = vanillaVae.encode(x)
        out, kld_loss = vanillaVae(x)
        sample = vanillaVae.sample(n_samples=2)

        print('***** Vanilla_VAE *****')
        print('Input:', x.shape)
        print('Encode:', z.shape)
        print('Output:', out.shape)
        print('KLD_Loss:', kld_loss.detach())
        print('Sample:', sample.shape)
    
    main()