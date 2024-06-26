import numpy as np
import os.path as osp
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import FashionMNIST


class FashionDataset(Dataset):

    dataset_dir = 'fashion'

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.prepare_data()

    def prepare_data(self):
        trainset = FashionMNIST(self.dataset_dir, download=True, train=True)

        testset = FashionMNIST(self.dataset_dir, download=True, train=False)

        self.dataset = ConcatDataset(datasets=[trainset, testset])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return np.array(self.dataset[index][0]), {
            'label': self.dataset[index][1]
        }


if __name__ == "__main__":
    dataset = FashionDataset(data_dir='data')
    print(len(dataset))
    image, cond = dataset[0]
    label = cond['label']
    print(image.shape, label)
    image.show()