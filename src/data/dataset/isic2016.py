import glob
import imageio
import numpy as np
import os.path as osp
from torch.utils.data import Dataset


class ISIC2016Dataset(Dataset):

    dataset_dir = "isic-2016"
    dataset_url = "https://challenge.isic-archive.com/data/#2016"

    def __init__(
        self,
        data_dir: str = 'data',
        train_val_test_dir: str = None,
    ) -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        if train_val_test_dir:
            self.dataset_dir = osp.join(self.dataset_dir, train_val_test_dir)
            self.img_paths = glob.glob(f"{self.dataset_dir}/*.jpg")
        else:
            img_dirs = [
                f"{self.dataset_dir}/ISBI2016_ISIC_Part3B_Test_Data/*.jpg",
                f"{self.dataset_dir}/ISBI2016_ISIC_Part3B_Training_Data/*.jpg",
            ]

            self.img_paths = [
                img_path for img_dir in img_dirs
                for img_path in glob.glob(img_dir)
            ]

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = img_path.replace(".jpg", "_Segmentation.png")

        image = imageio.v2.imread(img_path)
        mask = imageio.v2.imread(mask_path)

        return mask, {'image': image}


if __name__ == "__main__":
    dataset = ISIC2018Dataset(data_dir='data', train_val_test_dir=None)
    print(len(dataset))

    mask, cond = dataset[0]
    image = cond['image']

    print(image.shape, image.dtype, type(image))
    print(mask.shape, mask.dtype, type(mask))

    print(np.unique(mask))

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.show()

    print(mask.max())