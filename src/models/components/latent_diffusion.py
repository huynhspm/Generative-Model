from typing import Tuple, Optional

# import sys

# sys.path.insert(0, '')

import torch
import pyrootutils
import torch.nn as nn

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.autoencoder import AutoEncoder
from src.models.components import DiffusionModel


class LatentDiffusion(nn.Module):
    """
    ### Latent Diffusion
    """

    def __init__(
        self,
        autoencoder: AutoEncoder,
        diffusion_model: DiffusionModel,
    ) -> None:

        super().__init__()

        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model

    def autoencoder_encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        ### Get scaled latent space representation of the image
        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """
        return self.autoencoder.encode(image).sample()

    def autoencoder_decode(self, z: torch.Tensor):
        """
        ### Get image from the latent representation
        We scale down by the scaling factor and then decode.
        """
        return self.autoencoder.decode(z)

    def get_q_sample(self,
                     z: torch.Tensor,
                     noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ### get sample to train
        """
        return self.diffusion_model.get_q_sample(z, noise)

    def get_p_sample(self,
                     z: Optional[torch.Tensor],
                     num_sample: int = 9,
                     repeat_noise: bool = False) -> torch.Tensor:
        """
        ### inference 
        """
        return self.diffusion_model.get_p_sample(z, num_sample, repeat_noise)


if __name__ == "__main__":
    from src.models.components.denoise_models import UnetModel
    autoencoder = AutoEncoder(img_channels=3,
                              channels=64,
                              z_channels=4,
                              backbone="Resnet",
                              n_layer_blocks=2,
                              channel_multipliers=[1, 2, 4, 4],
                              attention_level=[0, 1, 2],
                              emb_channels=4)

    df = DiffusionModel(n_steps=5,
                        img_dims=[3, 64, 64],
                        denoise_model=UnetModel(
                            img_channels=3,
                            channels=64,
                            backbone='Resnet',
                            n_layer_blocks=2,
                            channel_multipliers=[1, 2, 4, 4],
                            attention_levels=[0, 1, 2],
                            n_attention_heads=4,
                            n_attention_layers=1))

    ldm = LatentDiffusion(autoencoder, df)

    input = torch.randn(2, 3, 128, 128)

    z = ldm.autoencoder_encode(input)
    print(z.shape)

    # t = torch.randint(5, [1])

    # targets, preds = df.get_sample(input)
    # print(targets.shape, preds.shape)

    # img = df.p_sample(input, t)
    # print(img.shape)