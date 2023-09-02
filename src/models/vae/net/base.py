from typing import Any, List, Tuple

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

    def sample(self,
               n_samples:int,
               device: str = 'cpu') -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(n_samples, 
                        self.latent_dims[0], 
                        self.latent_dims[1], 
                        self.latent_dims[2])

        z = z.to(device)
        samples = self.decode(z)
        return samples
    
    @abstractmethod
    def forward(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def loss_function(self, 
                      img: Tensor, 
                      recons_img: Tensor, 
                      **kwargs) -> Tensor:
        pass

class GaussianDistribution:
    """
    ## Gaussian Distribution
    """

    def __init__(self, parameters: Tensor) -> None:
        """
        parameters: are the means and log of variances of the embedding of shape
            `[batch_size, z_channels * 2, z_height, z_width]`
        """
        # Split mean and log of variance
        self.mean, self.log_var = torch.chunk(parameters, 2, dim=1)

        # Clamp the log of variances
        # self.log_var = torch.clamp(self.log_var, -30.0, 20.0)

        # Calculate standard deviation
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self) -> Tensor:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """

        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mean ** 2 - self.log_var.exp(), dim = [1, 2, 3]), dim = 0)
        
        # Sample from the distribution N(mean, std) = mean + std * N(0, 1)
        z = self.mean + self.std * torch.randn_like(self.std)

        # print('-' * 60)
        # print('mean:', self.mean.min().item(), self.mean.max().item(), '\n',
        #       'std:', self.std.min().item(), self.std.max().item(), '\n',
        #       'z:', z.min().item(), z.max().item())
        # print('-' * 60)

        return z, kld_loss
