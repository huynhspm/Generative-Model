from typing import List

import torch
import pyrootutils
from torch import nn
from torch import Tensor

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.up_down import Encoder


class ImageEmbedder(nn.Module):
    """
    ### Image Embedder module
    """

    def __init__(self,
                 img_channels: int,
                 d_embed: int = 3,
                 base_channels: int = 64,
                 block: str = "Residual",
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4],
                 attention: str = 'Attention',
                 autoencoder_weight_path: str = None,
                 freeze: bool = False) -> None:
        """_summary_

        Args:
            img_channels (int): is the number of channels in the image
            d_embed (int, optional): is the number of channels in the embedding space. Defaults to 3.
            base_channels (int, optional): is the number of channels in the first convolution layer. Defaults to 32.
            block (str, optional): is the block of block in each layers of embedder. Defaults to "Residual".
            n_layer_blocks (int, optional): is the number of resnet layers at each resolution. Defaults to 1.
            channel_multipliers (List[int], optional): are the multiplicative factors for the number of channels in the subsequent blocks. Defaults to [1, 2, 4].
            attention (str, optional): _description_. Defaults to 'Attention'.
            autoencoder_weight_path (str, optional): _description_. Defaults to None.
        """

        super().__init__()

        self.encoder = Encoder(in_channels=img_channels,
                               base_channels=base_channels,
                               z_channels=d_embed,
                               block=block,
                               n_layer_blocks=n_layer_blocks,
                               channel_multipliers=channel_multipliers,
                               attention=attention)

        if autoencoder_weight_path is not None:
            self.load_weights(autoencoder_weight_path, freeze)
        else:
            print('Image_Embedder will train with diffusion')

    def load_weights(self, autoencoder_weight_path: str, freeze: bool = False):
        # Load weights from a file
        autoencoder_state_dict = torch.load(
            autoencoder_weight_path)['state_dict']

        embedder_state_dict = dict()
        for key in autoencoder_state_dict.keys():
            if 'encoder' not in key or 'ema' in key: continue

            new_key = key.replace('net.encoder.', '')
            embedder_state_dict[new_key] = autoencoder_state_dict[key]

        self.encoder.load_state_dict(embedder_state_dict)

        if freeze:        
            self.encoder.eval().requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`

        Returns:
            Tensor: _description_
        """

        return self.encoder(x)


if __name__ == "__main__":
    x = torch.randn(2, 1, 128, 128)
    embedder = ImageEmbedder(img_channels=1,
                             base_channels=64,
                             d_embed=1,
                             autoencoder_weight_path="last.ckpt")
    out = embedder(x)

    print('***** Embedder *****')
    print('Input:', x.shape)
    print('Output:', out.shape)