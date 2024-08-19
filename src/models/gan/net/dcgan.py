from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch import Tensor

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.gan.net import GAN

class Generator(nn.Module):
    def __init__(self, 
                latent_dim: int, 
                img_channels: int, 
                img_size: int,
                d_cond_label: int | None = None) -> None:
        super().__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + (d_cond_label if d_cond_label is not None else 0), 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, cond: Dict[str, Tensor], z: Tensor) -> Tensor:
        if cond is not None:
            if 'label' in cond.keys():
                assert cond['label'].shape[0] == z.shape[0], 'shape not match'
                z = torch.cat((z, cond["label"]), dim=1)

        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, 
                img_channels: int, 
                img_size: int, 
                d_cond_label: int | None = None) -> None:
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_channels + (d_cond_label if d_cond_label is not None else 0), 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, cond: Dict[str, Tensor], img: Tensor) -> Tensor:
        if cond is not None:
            if 'label' in cond.keys():
                assert cond['label'].shape[0] == img.shape[0], 'shape not match'
                img = torch.cat((img, cond["label"][:, :, None, None]), dim=1)

        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

if __name__ == "__main__":
    latent_dim = 100
    img_channels = 1
    img_size = 32

    dcgan = GAN(gen = Generator(latent_dim, img_channels, img_size),
                disc= Discriminator(img_channels, img_size),
                latent_dim=latent_dim)

    image = dcgan.sample(cond=None, num_sample=10)
    print(image.shape)

    label = dcgan.classify(cond=None, image=image)
    print(label.shape)