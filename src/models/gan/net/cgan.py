from typing import Tuple, Dict

import math
import torch
import torch.nn as nn
from torch import Tensor
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.gan.net import GAN


class Generator(nn.Module):
    def __init__(self, 
                latent_dim: int, 
                img_dims: Tuple[int, int, int], 
                d_cond_label: int | None = None) -> None:
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + (d_cond_label if d_cond_label is not None else 0), 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(math.prod(img_dims))),
            nn.Tanh()
        )

        self.img_dims = img_dims

    def forward(self, cond: Dict[str, Tensor], z: Tensor) -> Tensor:
        if cond is not None:
            if 'label' in cond.keys():
                assert cond['label'].shape[0] == z.shape[0], 'shape not match'
                z = torch.cat((z, cond["label"]), dim=1)

        img = self.model(z)
        img = img.view(img.size(0), *self.img_dims)
        return img

class Discriminator(nn.Module):
    def __init__(self, 
                img_dims: Tuple[int, int, int], 
                d_cond_label: int | None = None) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(math.prod(img_dims)) + (d_cond_label if d_cond_label is not None else 0), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, cond: Dict[str, Tensor], img: Tensor) -> Tensor:
        img_flat = img.view(img.size(0), -1)

        if cond is not None:
            if 'label' in cond.keys():
                assert cond['label'].shape[0] == img_flat.shape[0], 'shape not match'
                img_flat = torch.cat((img_flat, cond["label"]), dim=1)

        validity = self.model(img_flat)
        return validity

class CGAN(GAN):
    def __init__(self,
                gen: nn.Module,
                disc: nn.Module,
                latent_dim: int,
                label_embedder: nn.Module = None,
                image_embedder: nn.Module = None,
                text_embedder: nn.Module = None,) -> None:
        super().__init__(gen, disc, latent_dim)

        self.label_embedder = label_embedder
        self.image_embedder = image_embedder
        self.text_embedder = text_embedder

    def get_label_embedding(self, label: torch.Tensor):
        return self.label_embedder(label)

    def get_image_embedding(self, image: torch.Tensor):
        return self.image_embedder(image)

    def get_text_embedding(self, text: torch.Tensor):
        return self.text_embedder(text)

    def get_cond_embedding(self, cond: Dict[str, Tensor]):
        if cond is not None:
            if self.label_embedder is not None:
                assert 'label' in cond.keys(
                ), "must specify label if and only if this model is label-conditional"

                cond['label'] = self.get_label_embedding(cond['label'])

            if self.image_embedder is not None:
                assert 'image' in cond.keys(
                ), "must specify image if and only if this model is image-conditional"

                cond['image'] = self.get_image_embedding(cond['image'])

            if 'text' in cond.keys():
                assert 'text' in cond.keys(
                ), "must specify text if and only if this model is text-conditional"

                cond['text'] = self.get_text_embedding(cond['text'])
        return cond

    def classify(self, cond: Dict[str, Tensor], image: Tensor) -> Tensor:
        cond_embedded = self.get_cond_embedding(cond.copy())
        return super().classify(cond_embedded, image)

    def sample(self, 
            cond: Dict[str, Tensor],
            num_sample: int = 1,
            device: torch.device = torch.device('cpu')) -> Tensor:

        cond_embedded = self.get_cond_embedding(cond.copy())
        return super().sample(cond_embedded, num_sample, device)

if __name__ == "__main__":
    from src.models.components.embeds import LabelEmbedder

    latent_dim = 100
    img_dims = [1, 32, 32]
    n_classes = 10
    d_cond_label = 10

    gan = CGAN(gen = Generator(latent_dim, img_dims, d_cond_label),
                disc = Discriminator(img_dims, d_cond_label),
                latent_dim=latent_dim,
                label_embedder=LabelEmbedder(n_classes, d_embed=d_cond_label))

    cond = {"label": torch.randint(0, 10, (10,))}
    image = gan.sample(cond=cond, num_sample=10)
    print(image.shape)

    label = gan.classify(cond=cond, image=image)
    print(label.shape)  