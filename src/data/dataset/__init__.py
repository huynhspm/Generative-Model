from .anime import AnimeDataset
from .celeba import CelebADataset
from .cifar10 import Cifar10Dataset
from .cvc_clinic import CVCClinicDataset
from .dogcat import DogCatDataset
from .edge_coco import EdgeCOCODataset
from .fashion import FashionDataset
from .ffhq import FFHQDataset
from .gender import GenderDataset
from .imagenet import ImageNetDataset
from .isic import ISICDataset
from .mnist import MnistDataset
from .sketch_celeba import SketchCelebADataset
from .sketch_coco import SketchCOCODataset

__datasets = {
    'anime': AnimeDataset,
    'celeba': CelebADataset,
    'cifar10': Cifar10Dataset,
    'cvc_clinic': CVCClinicDataset,
    'dogcat': DogCatDataset,
    'edge_coco': EdgeCOCODataset,
    'fashion': FashionDataset,
    'ffhq': FFHQDataset,
    'gender': GenderDataset,
    'imagenet': ImageNetDataset,
    'isic': ISICDataset,
    'mnist': MnistDataset,
    'sketch_celeba': SketchCelebADataset,
    'sketch_coco': SketchCOCODataset,
}


def init_dataset(name, **kwargs):
    """Initializes a dataset."""
    avai_datasets = list(__datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_datasets))
    return __datasets[name](**kwargs)