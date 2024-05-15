from typing import List, Tuple

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

    def __init__(self,
                 img_dims: int,
                 z_channels: int = 3,
                 base_channels: int = 64,
                 block: str = 'Residual',
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4],
                 attention: str = 'Attention') -> None:
        """_summary_

        Args:
            img_dims (int): the list of dimension of image.
            z_channels (int, optional): is the number of channels in the embedding space. Defaults to 3.
            base_channels (int, optional): is the number of channels in the first convolution layer. Defaults to 64.
            block (str, optional):  is the block of block in each layers. Defaults to 'Residual'.
            n_layer_blocks (int, optional): is the number of resnet layers at each resolution. Defaults to 1.
            channel_multipliers (List[int], optional): are the multiplicative factors for the number of channels in the subsequent blocks. Defaults to [1, 2, 4].
            attention (str, optional): _description_. Defaults to 'Attention'.
        """
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(in_channels=img_dims[0],
                               base_channels=base_channels,
                               z_channels=z_channels,
                               block=block,
                               n_layer_blocks=n_layer_blocks,
                               channel_multipliers=channel_multipliers,
                               attention=attention,
                               double_z=True)

        self.decoder = Decoder(out_channels=img_dims[0],
                               base_channels=base_channels,
                               z_channels=z_channels,
                               block=block,
                               n_layer_blocks=n_layer_blocks,
                               channel_multipliers=channel_multipliers,
                               attention=attention)

        self.latent_dims = [
            z_channels,
            int(img_dims[1] / (1 << (len(channel_multipliers) - 1))),
            int(img_dims[2] / (1 << (len(channel_multipliers) - 1)))
        ]

    def encode(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        """
        ### Encode images to z representation

        img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_width]`
        mean_var = self.encoder(img)
        mean, log_var = torch.chunk(mean_var, 2, dim=1)

        std = torch.exp(0.5 * log_var)
        z = mean + std * torch.randn_like(std)

        return z

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
        # print(cfg)
        autoencoder: AutoEncoder = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 3, 32, 32)

        z = autoencoder.encode(x)
        out, _ = autoencoder(x)

        print('***** AutoEncoder *****')
        print('Input:', x.shape)
        print('Encode:', z.shape)
        print('Output:', out.shape)

    main()