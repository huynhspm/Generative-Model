import glob
import imageio
import os.path as osp
from torch.utils.data import Dataset
import cv2


class SketchCOCODataset(Dataset):
    dataset_dir = 'sketchy_coco'
    dataset_url = 'https://github.com/sysu-imsl/SketchyCOCO?fbclid=IwAR26d967nBV8h1G9ll9fvlTJzAaIW2WE98PF3AdnI0t8dRnWHpasJLKptok'
    labels = [
        '2', '3', '4', '5', '10', '11', '17', '18', '19', '20', '21', '22',
        '24', '25'
    ]

    def __init__(self, data_dir: str = 'data') -> None:
        """
            data_dir:
        """
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.img_paths = glob.glob(f"{self.dataset_dir}/Object/GT/*/*/*.png")

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        name = img_path.split('/')
        label = self.labels.index(name[-2])
        sketch_path = osp.join(self.dataset_dir, 'Object/Sketch', name[-3],
                               name[-2], name[-1])

        image = imageio.v2.imread(img_path, pilmode='RGB')
        sketch = imageio.v2.imread(sketch_path, pilmode='L')
        sketch = cv2.GaussianBlur(sketch, (3, 3), None)
        sketch = cv2.erode(sketch, kernel=(3, 3), iterations=3)

        return image, {'label': label, 'image': sketch}


if __name__ == "__main__":
    dataset = SketchCOCODataset(data_dir='data')
    print(len(dataset))
    image, cond = dataset[0]
    label, sketch = cond['label'], cond['image']
    print(image.shape, label, sketch.shape)

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Target Image')

    plt.subplot(1, 2, 2)
    plt.imshow(sketch, cmap='gray')
    plt.title('Sketch Image')

    plt.show()