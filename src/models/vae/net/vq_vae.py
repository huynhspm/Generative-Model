from typing import List, Tuple, Dict

import torch
import pyrootutils
from torch import nn
from torch import Tensor
import torch.nn.functional as F

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vae.net import BaseVAE
from src.models.components.up_down import Encoder, Decoder


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(
            0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0),
                                       self.K,
                                       device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot,
                                         self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(
            latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(
            0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]


class VQVAE(BaseVAE):
    """
    ## AutoEncoder

    This consists of the encoder and decoder modules.
    """

    def __init__(self,
                 img_dims: int,
                 z_channels: int = 3,
                 base_channels: int = 64,
                 num_embeddings: int = 512,
                 beta: float = 0.25,
                 block: str = 'Residual',
                 n_layer_blocks: int = 1,
                 channel_multipliers: List[int] = [1, 2, 4],
                 attention: str = 'Attention') -> None:
        """_summary_

        Args:
            img_dims (int): _description_
            z_channels (int, optional): _description_. Defaults to 64.
            base_channels (int, optional): _description_. Defaults to 32.
            num_embeddings (int, optional): _description_. Defaults to 512.
            beta (float, optional): _description_. Defaults to 0.25.
            block (str, optional): _description_. Defaults to 'Residual'.
            n_layer_blocks (int, optional): _description_. Defaults to 1.
            channel_multipliers (List[int], optional): _description_. Defaults to [1, 2, 4].
            attention (str, optional): _description_. Defaults to 'Attention'.
        """
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels=img_dims[0],
                               base_channels=base_channels,
                               z_channels=z_channels,
                               block=block,
                               n_layer_blocks=n_layer_blocks,
                               channel_multipliers=channel_multipliers,
                               attention=attention)

        self.decoder = Decoder(out_channels=img_dims[0],
                               base_channels=base_channels,
                               z_channels=z_channels,
                               block=block,
                               n_layer_blocks=n_layer_blocks,
                               channel_multipliers=channel_multipliers,
                               attention=attention)

        self.vq_layer = VectorQuantizer(num_embeddings, z_channels, beta)
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
        # Get embeddings with shape `[batch_size, z_channels, z_height, z_width]`
        z = self.encoder(img)
        z, vq_loss = self.vq_layer(z)
        return z, vq_loss

    def decode(self, z: Tensor) -> Tensor:
        """
        ### Decode images from latent s

        z: is the z representation with shape `[batch_size, z_channels, z_height, z_width]`
        """

        # Decode the image of shape `[batch_size, img_channels, img_height, img_width]`
        return self.decoder(z)

    def forward(self, img: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        z, vq_loss = self.encode(img)
        loss = {"vq_loss": vq_loss}
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
                config_name="vq_vae.yaml")
    def main(cfg: DictConfig):
        # print(cfg)
        vq_vae: VQVAE = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 3, 32, 32)

        x_encoded, vq_loss = vq_vae.encode(x)
        out, vq_loss = vq_vae(x)
        sample = vq_vae.sample(n_samples=2)

        print('***** VQVAE *****')
        print('Input:', x.shape)
        print('Encode:', x_encoded.shape)
        print('VQ_Loss:', vq_loss)
        print('Output:', out.shape)
        print('Sample:', sample.shape)

    main()