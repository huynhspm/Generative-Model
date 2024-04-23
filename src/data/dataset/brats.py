import glob
import numpy as np
import os.path as osp
from torch.utils.data import Dataset


class BRATSDataset(Dataset):

    dataset_dir = 'brats-2020'
    dataset_url = 'https://www.cbica.upenn.edu/MICCAI_BraTS2020_TrainingData'

    def __init__(
        self,
        data_dir: str = 'data',
        train_val_test_dir: str = None,
    ) -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.train_val_test_dir = train_val_test_dir

        if train_val_test_dir:
            self.dataset_dir = osp.join(self.dataset_dir, train_val_test_dir)
            self.img_paths = glob.glob(
                f"{self.dataset_dir}/*/image_slice_*.npy")
        else:
            img_dirs = [
                f"{self.dataset_dir}/Train/*/image_slice_*.npy",
                f"{self.dataset_dir}/Val/*/image_slice_*.npy",
                f"{self.dataset_dir}/Test/*/image_slice_*.npy",
            ]

            self.img_paths = [
                img_path for img_dir in img_dirs
                for img_path in glob.glob(img_dir)
            ]

    def prepare_data(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        mask_path = self.img_paths[index].replace('image', 'mask')

        image = np.load(image_path)

        if image.max() > 0:
            image = image / image.max()
        
        if self.train_val_test_dir == "Test-without-label":
            mask = np.zeros_like(image[..., :1])
        else:   
            mask = np.load(mask_path)

        return mask, {'image': image}


if __name__ == "__main__":
    dataset = BRATSDataset(
        data_dir='data',
        train_val_test_dir='Train',
    )
    print(len(dataset))

    mask, cond = dataset[0]
    images = cond['image']

    print(images.shape, images.dtype, type(images))
    print(mask.shape, mask.dtype, type(mask))
    
    print(np.unique(mask))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 5, 1)
    plt.imshow(images[:, :, 0], cmap='gray')
    plt.title('T1')

    plt.subplot(1, 5, 2)
    plt.imshow(images[:, :, 1], cmap='gray')
    plt.title('T1CE')

    plt.subplot(1, 5, 3)
    plt.imshow(images[:, :, 2], cmap='gray')
    plt.title('T2')

    plt.subplot(1, 5, 4)
    plt.imshow(images[:, :, 3], cmap='gray')
    plt.title('FLAIR')

    plt.subplot(1, 5, 5)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.show()