from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

class GAN(nn.Module):

    def __init__(self, 
                gen: nn.Module,
                disc: nn.Module,
                latent_dim: int) -> None:
        super().__init__()

        self.gen = gen
        self.gen.apply(weights_init)

        self.disc = disc
        self.disc.apply(weights_init)

        self.latent_dim = latent_dim

    def classify(self, 
                cond: Dict[str, Tensor],
                image: Tensor) -> Tensor:

        return self.disc(cond, image)
    
    def sample(self,
            cond: Dict[str, Tensor],
            num_sample: int = 1,
            device: torch.device = torch.device("cpu")) -> Tensor:

        z = torch.randn(num_sample, self.latent_dim, device=device)
        image = self.gen(cond, z)

        return image

if __name__ == "__main__":
    import pyrootutils
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    from src.models.gan.net.cgan import Generator, Discriminator

    latent_dim = 100
    img_dims = [1, 32, 32]

    gan = GAN(gen = Generator(latent_dim, img_dims, d_cond_label=None),
            disc = Discriminator(img_dims, d_cond_label=None),
            latent_dim=latent_dim)

    image = gan.sample(cond=None, num_sample=10)
    print(image.shape)

    label = gan.classify(cond=None, image=image)
    print(label.shape)