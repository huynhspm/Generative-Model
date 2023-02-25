from .base import BaseBlock
from .resnet import ResnetBlock
from .inception import InceptionBlock
from .vgg import VGGBlock
from .dense import DenseBlock

Backbones = {
    "Base": BaseBlock,
    "Resnet": ResnetBlock,
    "Inception": InceptionBlock,
    "VGG": VGGBlock,
    "Dense": DenseBlock,
}


def init_backbone(name):
    """Initializes block of backbone"""
    avai_backbones = list(Backbones.keys())
    if name not in avai_backbones:
        raise ValueError('Invalid backbone name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_backbones))
    return Backbones[name]

def get_all_backbones():
    return tuple(Backbones.values())