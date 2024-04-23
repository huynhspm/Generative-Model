import glob
import imageio
import numpy as np
import os.path as osp
from torch.utils.data import Dataset


class CVCClinicDataset(Dataset):

    dataset_dir = 'cvc_clinic'
    dataset_url = 'https://www.kaggle.com/datasets/balraj98/cvcclinicdb'

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.img_paths = glob.glob(f"{self.dataset_dir}/Original/*.png")

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = img_path.replace('Original', 'Ground_Truth')

        image = imageio.v2.imread(img_path)
        mask = imageio.v2.imread(mask_path, pilmode='L')

        return mask, {'image': image}


if __name__ == "__main__":
    dataset = CVCClinicDataset(data_dir='data')
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
    plt.imshow(mask)
    plt.title('Mask')
    plt.show()