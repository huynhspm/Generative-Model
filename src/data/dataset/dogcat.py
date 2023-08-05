import glob
import imageio
import os.path as osp
from torch.utils.data import Dataset


class DogCatDataset(Dataset):

    dataset_dir = 'dogcat'
    dataset_url = 'https://www.kaggle.com/datasets/tongpython/cat-and-dog'

    def __init__(self,
                 data_dir: str = 'data') -> None:
        """
            data_dir:
            transforms:
        """
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        img_dir = [
            f"{self.dataset_dir}/test_set/cats/*.jpg",
            f"{self.dataset_dir}/test_set/dogs/*.jpg",
            f"{self.dataset_dir}/training_set/cats/*.jpg",
            f"{self.dataset_dir}/training_set/dogs/*.jpg",
            f"{self.dataset_dir}/train/cat/*.jpg",
            f"{self.dataset_dir}/train/dog/*.jpg",
            f"{self.dataset_dir}/train/wild/*.jpg",
            f"{self.dataset_dir}/val/cat/*.jpg",
            f"{self.dataset_dir}/val/dog/*.jpg",
            f"{self.dataset_dir}/val/wild/*.jpg",
        ]

        self.img_paths = glob.glob(img_dir[0]) + glob.glob(
            img_dir[1]) + glob.glob(img_dir[2]) + glob.glob(
                img_dir[3]) + glob.glob(img_dir[4]) + glob.glob(
                    img_dir[5]) + glob.glob(img_dir[6]) + glob.glob(img_dir[7])

    def prepare_data(self) -> None:
        import opendatasets as od
        od.download(self.dataset_url)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = imageio.v2.imread(img_path)
        
        label = img_path.split('/')[-2]
        if label in ['dog', 'dogs']: label = 0
        elif label in ['cat', 'cats']: label = 1
        else: label = 2

        return image, label
    
if __name__ == "__main__":
    dataset = DogCatDataset(data_dir='data')
    print(len(dataset))
    image, label = dataset[0]
    print(image.shape, label)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()