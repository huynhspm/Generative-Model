import glob
import imageio
import os.path as osp
from torch.utils.data import Dataset


class GenderDataset(Dataset):

    dataset_dir = 'gender'
    dataset_url = 'https://www.kaggle.com/datasets/yasserhessein/gender-dataset'
    labels = ['Male', 'Female']

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        img_dir = [
            f"{self.dataset_dir}/Test/Female/*.jpg",
            f"{self.dataset_dir}/Test/Male/*.jpg",
            f"{self.dataset_dir}/Validation/Female/*.jpg",
            f"{self.dataset_dir}/Validation/Male/*.jpg",
            f"{self.dataset_dir}/Train/Female/*.jpg",
            f"{self.dataset_dir}/Train/Male/*.jpg"
        ]

        self.img_paths = glob.glob(img_dir[0]) + glob.glob(
            img_dir[1]) + glob.glob(img_dir[2]) + glob.glob(
                img_dir[3]) + glob.glob(img_dir[4]) + glob.glob(img_dir[5])

    def prepare_data(self) -> None:
        import opendatasets as od
        od.download(self.dataset_url)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = imageio.v2.imread(img_path)
        label = self.labels.index(img_path.split('/')[-2])

        return image, {'label': label}


if __name__ == "__main__":
    dataset = GenderDataset(data_dir='data')
    print(len(dataset))
    image, cond = dataset[0]
    label = cond['label']
    print(image.shape, label)

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()