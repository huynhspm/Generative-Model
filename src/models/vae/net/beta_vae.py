from typing import Tuple, Dict

import torch
import pyrootutils
from torch import Tensor
from torch.nn import functional as F

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.vae.net import VanillaVAE
from src.models.components.up_down import Encoder, Decoder


class BetaVAE(VanillaVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(
        self,
        latent_dims: Tuple[int, int, int],
        encoder: Encoder,
        decoder: Decoder,
        kld_weight: float = 1.0,
        beta: int = 4,
        gamma:float = 1000.,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type:str = 'B',
    ) -> None:
        """_summary_

        Args:
            latent_dims (Tuple[int, int, int]): _description_
            encoder (Encoder): _description_
            decoder (Decoder): _description_
            kld_weight (float, optional): _description_. Defaults to 1.0.
            beta (int, optional): _description_. Defaults to 4.
            gamma (float, optional): _description_. Defaults to 1000..
            max_capacity (int, optional): _description_. Defaults to 25.
            Capacity_max_iter (int, optional): _description_. Defaults to 1e5.
            loss_type (str, optional): _description_. Defaults to 'B'.
        """

        super().__init__(latent_dims, encoder, decoder, kld_weight)

        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

    def forward(self, img: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        self.num_iter += 1
        z, kld_loss = self.encode(img)

        if self.loss_type == 'H':
          loss = {"kld_loss": self.beta * self.kld_weight * kld_loss}
        elif self.loss_type == 'B':
          self.C_max = self.C_max.to(img.device)
          C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
          loss = {"kld_loss": self.gamma * self.kld_weight* (kld_loss - C).abs()}
        else:
            raise ValueError('Undefined loss type.')
        
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
                config_name="beta_vae.yaml")
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

        beta_vae: BetaVAE = hydra.utils.instantiate(cfg)
        x = torch.randn(2, 3, 32, 32)

        z, kld_loss = beta_vae.encode(x)
        output, kld_loss = beta_vae(x)
        sample = beta_vae.sample(n_samples=2)

        print('***** VanillaVAE *****')
        print('Input:', x.shape)
        print('Encode:', z.shape)
        print('KLD_Loss:', kld_loss)
        print('Output:', output.shape)
        print('Sample:', sample.shape)

    main()