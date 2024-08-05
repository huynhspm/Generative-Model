from typing import Tuple, Dict

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

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
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

    def __init__(
        self,
        latent_dims: Tuple[int, int, int],
        vq_layer: VectorQuantizer,
        encoder: Encoder,
        decoder: Decoder,
    ) -> None:
        """_summary_

        Args:
            latent_dims (Tuple[int, int, int]): _description_
            vq_layer (VectorQuantizer): _description_
            encoder (Encoder): _description_
            decoder (Decoder): _description_
        """

        super().__init__()

        self.latent_dims = latent_dims
        self.vq_layer = vq_layer
        self.encoder = encoder
        self.decoder = decoder
        self.vq_layer = vq_layer
        

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
        cfg["encoder"]["z_channels"] = 3
        cfg["decoder"]["z_channels"] = 3
        cfg["decoder"]["base_channels"] = 64
        cfg["decoder"]["block"] = "Residual"
        cfg["decoder"]["n_layer_blocks"] = 1
        cfg["decoder"]["drop_rate"] = 0.
        cfg["decoder"]["attention"] = "Attention"
        cfg["decoder"]["channel_multipliers"] = [1, 2, 3]
        cfg["decoder"]["n_attention_heads"] = None
        cfg["decoder"]["n_attention_layers"] = None
        cfg["vq_layer"]["embedding_dim"] = 3
        print(cfg)
        
        vq_vae: VQVAE = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 3, 32, 32)

        z, vq_loss = vq_vae.encode(x)
        output, vq_loss = vq_vae(x)
        sample = vq_vae.sample(n_samples=2)

        print('***** VQVAE *****')
        print('Input:', x.shape)
        print('Encode:', z.shape)
        print('VQ_Loss:', vq_loss)
        print('Decode:', output.shape)
        print('Sample:', sample.shape)

    main()