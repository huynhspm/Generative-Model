import torch
import pyrootutils
from torch import nn

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.models.components.encoder import Encoder
from src.models.components.decoder import Decoder

class AutoEncoder(nn.Module):
    """
    ## AutoEncoder

    This consists of the encoder and decoder modules.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        emb_channels: int = 4,
        z_channels: int = 4,
    ) -> None:
        """
        encoder:
        decoder:
        emb_channels: is the number of dimensions in the quantized embedding space
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # Convolution to map from embedding space to quantized embedding space moments
        self.quant_conv = nn.Conv2d(in_channels=2 * z_channels,
                                    out_channels=2 * emb_channels,
                                    kernel_size=1)

        # Convolution to map from quantized embedding space back to embedding space
        self.post_quant_conv = nn.Conv2d(in_channels=emb_channels,
                                         out_channels=z_channels,
                                         kernel_size=1)

    def encode(self, img: torch.Tensor) -> 'GaussianDistribution':
        """
        ### Encode images to latent representation

        img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
        z = self.encoder(img)

        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)

        # Return the distribution
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        ### Decode images from latent representation

        z: is the latent representation with shape `[batch_size, emb_channels, z_height, z_height]`
        """
        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)

        # Decode the image of shape `[batch_size, channels, height, width]`
        return self.decoder(z)

    def forward(self, img: torch.Tensor):
        z = self.encode(img).sample()
        return self.decode(z)


class GaussianDistribution:
    """
    ## Gaussian Distribution
    """

    def __init__(self, parameters: torch.Tensor) -> None:
        """
        parameters: are the means and log of variances of the embedding of shape
            `[batch_size, z_channels * 2, z_height, z_height]`
        """
        # Split mean and log of variance
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)

        # Clamp the log of variances
        self.log_var = torch.clamp(log_var, -30.0, 20.0)

        # Calculate standard deviation
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self) -> torch.Tensor:
        # Sample from the distribution
        return self.mean + self.std * torch.randn_like(self.std)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "net")
    
    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="autoencoder.yaml")
    def main(cfg: DictConfig):
        cfg.decoder.out_channels = 1
        autoencoder: AutoEncoder = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 1, 32, 32)
        print('input:', x.shape)
        print('output encoder:', autoencoder.encode(x).sample().shape)
        print('output autoencoder', autoencoder(x).shape)

    main()