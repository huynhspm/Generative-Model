import glob
import numpy as np
import os.path as osp
from torch.utils.data import Dataset


class LIDCDataset(Dataset):

    dataset_dir = 'lidc'
    dataset_url = 'https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254'

    def __init__(
        self,
        data_dir: str = 'data',
        multi_mask: bool = False,
        train_val_test_dir: str = None,
    ) -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir, "Multi-Annotations")
        if train_val_test_dir:
            self.dataset_dir = osp.join(self.dataset_dir, train_val_test_dir)
            self.img_paths = glob.glob(f"{self.dataset_dir}/Image/*/slice_*.npy")
        else:
            img_dirs = [
                f"{self.dataset_dir}/Train/Image/*/slice_*.npy",
                f"{self.dataset_dir}/Val/Image/*/slice_*.npy",
                f"{self.dataset_dir}/Test/Image/*/slice_*.npy",
            ]

            self.img_paths = [
                img_path for img_dir in img_dirs
                for img_path in glob.glob(img_dir)
            ]

        self.multi_mask = multi_mask

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = img_path.replace('Image', 'Mask').replace('.npy', '_e.npy')

        image = np.load(img_path)
        mask = np.load(mask_path)

        image = (image - image.min()) / (image.max() - image.min())
        mask = mask.astype(np.float64)

        if not self.multi_mask:
            return mask, {'image': image}

        # multi mask
        masks = [np.load(mask_path.replace('_e', f'_{i}')).astype(np.float64) for i in range(4)]
        masks = np.stack(masks, axis=-1)

        return mask, {'image': image, 'masks': masks}


if __name__ == "__main__":
    dataset = LIDCDataset(data_dir='data',
                          multi_mask=True)
    print(len(dataset))

    mask, cond = dataset[0]
    image = cond['image']
    masks = cond['masks']
    
    print(image.shape, image.dtype, type(image))
    print(mask.shape, mask.dtype, type(mask))

    print(np.unique(mask))

    variance = masks.var(axis=-1)
    print(masks.shape, variance.shape)

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))

    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Image')
    axs[0, 0].axis("off")

    axs[0, 1].imshow(mask, cmap='gray')
    axs[0, 1].set_title('Mask_e')
    axs[0, 1].axis("off")

    axs[0, 2].imshow(masks[:, :, 0], cmap='gray')
    axs[0, 2].set_title('Mask_0')
    axs[0, 2].axis("off")

    axs[1, 0].imshow(masks[:, :, 1], cmap='gray')
    axs[1, 0].set_title('Mask_1')
    axs[1, 0].axis("off")

    axs[1, 1].imshow(masks[:, :, 2], cmap='gray')
    axs[1, 1].set_title('Mask_2')
    axs[1, 1].axis("off")

    axs[1, 2].imshow(masks[:, :, 3], cmap='gray')
    axs[1, 2].set_title('Mask_3')
    axs[1, 2].axis("off")

    axs[2, 0] = sns.heatmap(variance)
    axs[2, 0].set_title('Variance')
    axs[2, 0].axis("off")

    plt.show()

    multi_masks = [dataset[i][1]['masks'] for i in range(len(dataset))]
    multi_masks = np.stack(multi_masks, axis=0)
    print(multi_masks.shape)

    variance = (multi_masks / 255.0).var(axis=-1)
    print(variance.shape, variance.min(), variance.max())

    print('Mean variance:', variance.mean())
    ids = variance > 0
    print('Mean boundary variance::', ((variance.sum(axis=(1, 2)) + 1) /
                                       (ids.sum(axis=(1, 2)) + 1)).mean())
