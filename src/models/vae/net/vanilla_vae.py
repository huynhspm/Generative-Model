from typing import Tuple, Dict

import torch
import pyrootutils
from torch import Tensor
from torch.nn import functional as F

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vae.net import BaseVAE
from src.models.components.up_down import Encoder, Decoder


class GaussianDistribution:
    """
    ## Gaussian Distribution
    """

    def __init__(self, parameters: Tensor) -> None:
        """
        parameters: are the means and log of variances of the embedding of shape
            `[batch_size, z_channels * 2, z_height, z_width]`
        """
        # Split mean and log of variance
        self.mean, self.log_var = torch.chunk(parameters, 2, dim=1)

        # Clamp the log of variances
        # self.log_var = torch.clamp(self.log_var, -30.0, 20.0)

        # Calculate standard deviation
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self) -> Tensor:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """

        kld_loss = torch.mean(
            -0.5 *
            torch.sum(1 + self.log_var - self.mean**2 - self.log_var.exp(),
                      dim=[1, 2, 3]),
            dim=0)

        # Sample from the distribution N(mean, std) = mean + std * N(0, 1)
        z = self.mean + self.std * torch.randn_like(self.std)

        return z, kld_loss


class VanillaVAE(BaseVAE):

    def __init__(
        self,
        latent_dims: Tuple[int, int, int],
        encoder: Encoder,
        decoder: Decoder,
        kld_weight: float = 1.0,
    ) -> None:
        """_summary_

        Args:
            latent_dims (Tuple[int, int, int]): _description_
            encoder (Encoder): _description_
            decoder (Decoder): _description_
            kld_weight (float, optional): _description_. Defaults to 1.0.
        """
        
        super().__init__()

        self.latent_dims = latent_dims
        self.kld_weight = kld_weight
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        """
        ### Encode images to z representation

        img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_width]`
        mean_var = self.encoder(img)
        z, kld_loss = GaussianDistribution(mean_var).sample()
        return z, kld_loss

    def decode(self, z: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        ### Decode images from latent s

        z: is the z representation with shape `[batch_size, z_channels, z_height, z_width]`
        """

        # Decode the image of shape `[batch_size, img_channels, img_height, img_width]`
        return self.decoder(z)

    def forward(self, img: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        z, kld_loss = self.encode(img)
        loss = {"kld_loss": self.kld_weight * kld_loss}
        return self.decode(z), loss


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
        cfg["encoder"]["z_channels"] = 3
        cfg["decoder"]["out_channels"] = 3
        cfg["decoder"]["z_channels"] = 3
        cfg["decoder"]["base_channels"] = 64
        cfg["decoder"]["block"] = "Residual"
        cfg["decoder"]["n_layer_blocks"] = 1
        cfg["decoder"]["drop_rate"] = 0.
        cfg["decoder"]["attention"] = "Attention"
        cfg["decoder"]["channel_multipliers"] = [1, 2, 3]
        cfg["decoder"]["n_attention_heads"] = None
        cfg["decoder"]["n_attention_layers"] = None
        print(cfg)

        vanilla_vae: VanillaVAE = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 3, 32, 32)

        z, kld_loss = vanilla_vae.encode(x)
        output, kld_loss = vanilla_vae(x)
        sample = vanilla_vae.sample(n_samples=2)

        print('***** VanillaVAE *****')
        print('Input:', x.shape)
        print('Encode:', z.shape)
        print('KLD_Loss:', kld_loss)
        print('Decode:', output.shape)
        print('Sample:', sample.shape)

    main()