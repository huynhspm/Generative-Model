import os
import cv2
import torch
import nibabel
import numpy as np
from torch.utils.data import Dataset


class BRATSDataset(Dataset):

    dataset_dir = 'brats'
    dataset_url = 'https://www.cbica.upenn.edu/MICCAI_BraTS2020_TrainingData'
    seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = os.path.join(data_dir, self.dataset_dir,
                                        'MICCAI_BraTS2020_TrainingData')

        self.database = []
        for root, dirs, files in os.walk(self.dataset_dir):
            if not dirs:
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3].split('.')[0]

                    datapoint[seqtype] = os.path.join(root, f)
                self.database.append(datapoint)

    def prepare_data(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.database) * 155

    def __getitem__(self, index):
        n = index // 155
        slice = index % 155
        filedict = self.database[n]
        target_size = (64, 64)

        datapoint = []
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            img = np.array(nib_img.get_fdata())[:, :, slice]
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            datapoint.append(torch.tensor(img))
        datapoint = torch.stack(datapoint, dim=0)

        images = datapoint[:-1, ...].to(torch.float32)
        mask = torch.where(datapoint[-1, ...] > 0, 1, 0).to(torch.float32)

        return mask, {'image': images}


if __name__ == "__main__":
    dataset = BRATSDataset(data_dir='data')
    print(len(dataset))
    mask, cond = dataset[80]
    images = cond['image']
    print(images.shape, mask.shape)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 5, 1)
    plt.imshow(images[0] / images[0].max(), cmap='gray')
    plt.title('T1')

    plt.subplot(1, 5, 2)
    plt.imshow(images[1] / images[1].max(), cmap='gray')
    plt.title('T1CE')

    plt.subplot(1, 5, 3)
    plt.imshow(images[2] / images[2].max(), cmap='gray')
    plt.title('T2')

    plt.subplot(1, 5, 4)
    plt.imshow(images[3] / images[3].max(), cmap='gray')
    plt.title('FLAIR')

    plt.subplot(1, 5, 5)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.show()