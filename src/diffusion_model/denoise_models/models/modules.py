import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_model.denoise_models.attentions.self_attention_wrapper import SAWrapper
from diffusion_model.denoise_models.attentions.spatial_transformer import SpatialTransformer
from diffusion_model.denoise_models.backbones.base import BaseBlock
from diffusion_model.denoise_models.backbones.vgg import VGGBlock
from diffusion_model.denoise_models.backbones.resnet import ResnetBlock


class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of 2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)


class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs
    This sequential module can compose of different modules suck as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self,
                x: torch.Tensor,
                t_emb: torch.Tensor,
                cond: torch.Tensor = None):
        for layer in self:
            if isinstance(layer, (BaseBlock, VGGBlock, ResnetBlock)):
                x = layer(x, t_emb)
            elif isinstance(layer, (SpatialTransformer, SAWrapper)):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x