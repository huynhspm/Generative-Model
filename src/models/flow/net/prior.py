import torch
import torch.nn.functional as F
from torch.distributions import Distribution

class LogisticDistribution(Distribution):

    arg_constraints = {}

    def __init__(self):
        super().__init__()

    def log_prob(self, x):
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size):

        # uniform distribution
        z = torch.distributions.Uniform(0., 1.).sample(size)

        return torch.log(z) - torch.log(1. - z)