import glob
import imageio
import os.path as osp
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):

    dataset_dir = 'imagenet'
    dataset_url = ''

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        img_dirs = [
            f"{self.dataset_dir}/test/images/*.JPEG",
            f"{self.dataset_dir}/train/*/images/*.JPEG",
            f"{self.dataset_dir}/val/images/*.JPEG",
        ]

        self.img_paths = [
            img_path for img_dir in img_dirs for img_path in glob.glob(img_dir)
        ]

    def prepare_data(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index):
        # no label
        img_path = self.img_paths[index]
        image = imageio.v2.imread(img_path)

        return image, {'label': -1}
