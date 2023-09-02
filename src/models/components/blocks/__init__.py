from .base_block import BaseBlock
from .residual_block import ResidualBlock
from .inception_block import InceptionBlock
from .vgg_block import VGGBlock
from .dense_block import DenseBlock

Blocks = {
    "Base": BaseBlock,
    "Residual": ResidualBlock,
    "Inception": InceptionBlock,
    "VGG": VGGBlock,
    "Dense": DenseBlock,
}


def init_block(name):
    """Initializes block of backbone"""
    avai_backbones = list(Blocks.keys())
    if name not in avai_backbones:
        raise ValueError('Invalid backbone name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_backbones))
    return Blocks[name]

def get_all_blocks():
    return tuple(Blocks.values())