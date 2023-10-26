import glob
import imageio
import os.path as osp
from torch.utils.data import Dataset


class AnimeDataset(Dataset):

    dataset_dir = 'anime'
    dataset_url = 'anime'

    def __init__(self,
                 data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.img_paths = glob.glob(f"{self.dataset_dir}/*.jpg")

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = imageio.v2.imread(img_path)
        return image, -1
    
if __name__ == "__main__":
    dataset = AnimeDataset(data_dir='data')
    print(len(dataset))
    image, label = dataset[0]
    print(image.shape, label)
    
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()