from typing import Tuple

import torch
import torch.nn as nn
import pyrootutils
from torch import Tensor
import torch.nn.functional as F

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.up_down import Encoder, Decoder


class AutoEncoder(nn.Module):
    """
    ## AutoEncoder

    This consists of the encoder and decoder modules.
    """

    def __init__(
        self,
        latent_dims: Tuple[int, int, int],
        encoder: Encoder,
        decoder: Decoder,
    ) -> None:
        """_summary_

        Args:
            latent_dims (List[int, int, int]): _description_
            encoder (Encoder): _description_
            decoder (Decoder): _description_
        """
        
        super().__init__()

        self.latent_dims = latent_dims
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        """
        ### Encode images to z representation

        img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Get embeddings with shape `[batch_size, z_channels, z_height, z_width]`
        return self.encoder(img)

    def decode(self, z: Tensor) -> Tensor:
        """
        ### Decode images from latent s

        z: is the z representation with shape `[batch_size, z_channels, z_height, z_width]`
        """

        # Decode the image of shape `[batch_size, img_channels, img_height, img_width]`
        return self.decoder(z)

    def forward(self, img: Tensor) -> Tuple[Tensor, None]:
        z = self.encode(img)
        return self.decode(z), None


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

        autoencoder: AutoEncoder = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 3, 32, 32)

        z = autoencoder.encode(x)
        output, _ = autoencoder(x)

        print('***** AutoEncoder *****')
        print('Input:', x.shape)
        print('Encode:', z.shape)
        print('Decode:', output.shape)

    main()