from .simple_net import SimpleBlock
from .res_net import ResnetBlock
from .inception_net import InceptionBlock
from .vgg_net import VGGBlock
from .dense_net import DenseBlock

Backbones = {
    "Base": SimpleBlock,
    "ResNet": ResnetBlock,
    "InceptionNet": InceptionBlock,
    "VGGNet": VGGBlock,
    "DenseNet": DenseBlock,
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