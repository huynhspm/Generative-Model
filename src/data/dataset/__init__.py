from .celeba import CelebADataset
from .cifar10 import Cifar10Dataset
from .dogcat import DogCatDataset
from .fashion import FashionDataset
from .gender import GenderDataset
from .imagenet import ImageNetDataset
from .mnist import MnistDataset

__datasets = {
    'celeba': CelebADataset,
    'cifar10': Cifar10Dataset,
    'dogcat': DogCatDataset,
    'fashion': FashionDataset,
    'gender': GenderDataset,
    'imagenet': ImageNetDataset,
    'mnist': MnistDataset,
}


def init_dataset(name, **kwargs):
    """Initializes a dataset."""
    avai_datasets = list(__datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_datasets))
    return __datasets[name](**kwargs)