import glob
import imageio
import os.path as osp
from torch.utils.data import Dataset


class AFHQDataset(Dataset):

    dataset_dir = 'afhq'
    dataset_url = 'https://www.kaggle.com/datasets/andrewmvd/animal-faces'
    labels = ['dog', 'cat', 'wild']

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        img_dirs = [
            f"{self.dataset_dir}/train/cat/*.jpg",
            f"{self.dataset_dir}/train/dog/*.jpg",
            f"{self.dataset_dir}/train/wild/*.jpg",
            f"{self.dataset_dir}/val/cat/*.jpg",
            f"{self.dataset_dir}/val/dog/*.jpg",
            f"{self.dataset_dir}/val/wild/*.jpg",
        ]

        self.img_paths = [
            img_path for img_dir in img_dirs for img_path in glob.glob(img_dir)
        ]

    def prepare_data(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = imageio.v2.imread(img_path)
        label = self.labels.index(img_path.split('/')[-2])

        return image, {'label': label}


if __name__ == "__main__":
    dataset = AFHQDataset(data_dir='data')
    print(len(dataset))
    image, cond = dataset[0]
    label = cond['label']
    print(image.shape, label)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()