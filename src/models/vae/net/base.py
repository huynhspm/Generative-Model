from typing import Tuple

import torch
from torch import nn
from torch import Tensor
from abc import abstractmethod


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def decode(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, n_samples: int, device: str = 'cpu') -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(n_samples, self.latent_dims[0], self.latent_dims[1],
                        self.latent_dims[2])

        z = z.to(device)
        samples = self.decode(z)
        return samples

    @abstractmethod
    def forward(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def loss_function(self, img: Tensor, recons_img: Tensor,
                      **kwargs) -> Tensor:
        pass