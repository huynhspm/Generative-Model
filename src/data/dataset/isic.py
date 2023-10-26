import glob
import imageio
import os.path as osp
from torch.utils.data import Dataset


class ISICDataset(Dataset):

    dataset_dir = 'isic'
    dataset_url = 'https://challenge.isic-archive.com/data/'

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        img_dir = [
            f"{self.dataset_dir}/ISBI2016_ISIC_Part3B_Test_Data/*.jpg",
            f"{self.dataset_dir}/ISBI2016_ISIC_Part3B_Training_Data/*.jpg"
        ]

        self.img_paths = glob.glob(img_dir[0]) + glob.glob(img_dir[1])

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = img_path[0:-4] + '_Segmentation.png'
        image = imageio.v2.imread(img_path)
        mask = imageio.v2.imread(mask_path)

        return image, mask


if __name__ == "__main__":
    dataset = ISICDataset(data_dir='data')
    print(len(dataset))
    image, mask = dataset[0]
    print(image.shape, mask.shape)

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title('Mask')
    plt.show()