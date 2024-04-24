from .afhq import AFHQDataset
from .anime import AnimeDataset
from .brats2020 import BraTS2020Dataset
from .brats2021 import BraTS2021Dataset
from .celeba import CelebADataset
from .cifar10 import Cifar10Dataset
from .cvc_clinic import CVCClinicDataset
from .edge_coco import EdgeCOCODataset
from .fashion import FashionDataset
from .ffhq import FFHQDataset
from .gender import GenderDataset
from .imagenet import ImageNetDataset
from .isic2016 import ISIC2016Dataset
from .isic2018 import ISIC2018Dataset
from .lidc import LIDCDataset
from .mnist import MnistDataset
from .sketch_celeba import SketchCelebADataset
from .sketch_coco import SketchCOCODataset

__datasets = {
    'afhq': AFHQDataset,
    'anime': AnimeDataset,
    'brats2020': BraTS2020Dataset,
    'brats2021': BraTS2021Dataset,
    'celeba': CelebADataset,
    'cifar10': Cifar10Dataset,
    'cvc_clinic': CVCClinicDataset,
    'edge_coco': EdgeCOCODataset,
    'fashion': FashionDataset,
    'ffhq': FFHQDataset,
    'gender': GenderDataset,
    'imagenet': ImageNetDataset,
    'isic2016': ISIC2016Dataset,
    'isic2018': ISIC2018Dataset,
    'lidc': LIDCDataset,
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