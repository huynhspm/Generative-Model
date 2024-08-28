import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Uniform


class NICE(nn.Module):
    num_net_layers = 6
    num_hidden_units = 1000

    def __init__(self, img_dims, num_coupling_layers=3):
        super().__init__()

        self.img_dims = img_dims
        self.data_dim = math.prod(img_dims)

        # alternating mask orientations for consecutive coupling layers
        masks = [self._get_mask(self.data_dim, orientation=(i % 2 == 0))
                                    for i in range(num_coupling_layers)]

        self.coupling_layers = nn.ModuleList([CouplingLayer(data_dim=self.data_dim,
                                    hidden_dim=self.num_hidden_units,
                                    mask=masks[i], num_layers=self.num_net_layers)
                                    for i in range(num_coupling_layers)])

        self.scaling_layer = ScalingLayer(data_dim=self.data_dim)

        self.prior = LogisticDistribution()

    def forward(self, x):
        x = x.view(x.shape[0], -1) # flatten
        z, log_det_jacobian = self.f(x)
        log_likelihood = torch.sum(self.prior.log_prob(z), dim=1) + log_det_jacobian
        nll = -torch.mean(log_likelihood)
        return z, nll

    def f(self, x):
        z = x
        log_det_jacobian = 0
        for i, coupling_layer in enumerate(self.coupling_layers):
            z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
        z, log_det_jacobian = self.scaling_layer(z, log_det_jacobian)
        return z, log_det_jacobian

    def f_inverse(self, z):
        x = z
        x, _ = self.scaling_layer(x, 0, invert=True)
        for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
            x, _ = coupling_layer(x, 0, invert=True)
        return x

    def sample(self, num_samples, device: torch.device = torch.device("cpu")):
        z = self.prior.sample(size=[num_samples, self.data_dim], 
                            device=device)

        x = self.f_inverse(z)
        x = x.view(x.size(0), *self.img_dims)

        return x

    def _get_mask(self, dim, orientation=True):
        mask = torch.zeros(dim, dtype=torch.float32)
        mask[::2] = 1.
        if orientation:
            mask = 1. - mask     # flip mask orientation

        return mask


class CouplingLayer(nn.Module):
    """
    Implementation of the additive coupling layer from section 3.2 of the NICE
    paper.
    """

    def __init__(self, data_dim, hidden_dim, mask, num_layers=4):
        super().__init__()

        assert data_dim % 2 == 0

        self.mask = mask

        modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Linear(hidden_dim, data_dim))

        self.m = nn.Sequential(*modules)

    def forward(self, x, logdet, invert=False):
        self.mask = self.mask.to(x.device)
        if not invert:
            x1, x2 = self.mask * x, (1. - self.mask) * x
            y1, y2 = x1, x2 + (self.m(x1) * (1. - self.mask))
            return y1 + y2, logdet

        # Inverse additive coupling layer
        y1, y2 = self.mask * x, (1. - self.mask) * x
        x1, x2 = y1, y2 - (self.m(y1) * (1. - self.mask))
        return x1 + x2, logdet


class ScalingLayer(nn.Module):
    """
    Implementation of the scaling layer from section 3.3 of the NICE paper.
    """
    def __init__(self, data_dim):
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.randn(1, data_dim, requires_grad=True))

    def forward(self, x, logdet, invert=False):
        log_det_jacobian = torch.sum(self.log_scale_vector)

        if invert:
            return torch.exp(- self.log_scale_vector) * x, logdet - log_det_jacobian

        return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian


class LogisticDistribution(Distribution):

    arg_constraints = {}

    def __init__(self):
        super().__init__()

    def log_prob(self, x):
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size, device: torch.device = torch.device("cpu")):

        # uniform distribution
        z = torch.rand(size, device=device)

        return torch.log(z) - torch.log(1. - z)

if __name__ == "__main__":
    nice = NICE(img_dims=[1, 32, 32], num_coupling_layers=4)

    x = torch.randn(1, 1, 32, 32)
    z, likelihood = nice(x)
    samples = nice.sample(num_samples=2)

    print(z.shape, likelihood)
    print(samples.shape)

