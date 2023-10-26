import torch
import os.path as osp
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import MNIST


class MnistDataset(Dataset):

    dataset_dir = 'mnist'

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.prepare_data()

    def prepare_data(self) -> None:
        trainset = MNIST(self.dataset_dir, download=True, train=True)

        testset = MNIST(self.dataset_dir, download=True, train=False)

        self.dataset = ConcatDataset(datasets=[trainset, testset])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> torch.Tensor:
        return self.dataset[index]


if __name__ == "__main__":
    dataset = MnistDataset(data_dir='data')
    print(len(dataset))
    image, label = dataset[0]
    print(image, label)
    image.show()