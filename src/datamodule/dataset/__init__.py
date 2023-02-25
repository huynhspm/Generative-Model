from .cifar_dataset import CifarDataset
from .dogcat_dataset import DogCatDataset
from .fashion_dataset import FashionDataset
from .gender_dataset import GenderDataset
from .imagenet_dataset import ImageNetDataset
from .mnist_dataset import MnistDataset

__datasets = {
    'cifar': CifarDataset,
    'dogcat': DogCatDataset,
    'fashion': FashionDataset,
    'gender': GenderDataset,
    'imagenet': ImageNetDataset,
    'mnist': MnistDataset
}


def init_dataset(name, **kwargs):
    """Initializes a dataset."""
    avai_datasets = list(__datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_datasets))
    return __datasets[name](**kwargs)