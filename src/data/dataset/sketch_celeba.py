import glob
import imageio
import os.path as osp
from torch.utils.data import Dataset
import cv2


def get_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    edges = cv2.Canny(gray_blurred, threshold1=30, threshold2=100)
    sketch = cv2.bitwise_not(edges)

    sketch = cv2.erode(sketch, kernel=(3, 3), iterations=5)
    sketch = cv2.dilate(sketch, kernel=(3, 3), iterations=2)

    return sketch


class SketchCelebADataset(Dataset):
    dataset_dir = 'sketch_celeba'
    dataset_url = 'https://drive.google.com/drive/folders/1TxsSzPhZsJNijIXPINv05IUWhG3vBU-X?fbclid=IwAR3o4qVWRF16Q1FJRivhYsXaUZf7nFcobqz04d4-TsR3dPWtXqjL_Gl_wCg'

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.img_paths = glob.glob(f"{self.dataset_dir}/images/*.jpg")

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = imageio.v2.imread(img_path)
        # sketch = get_sketch(image)

        name = img_path.split('/')
        sketch_path = osp.join(self.dataset_dir, 'sketch', name[-1])
        sketch = imageio.v2.imread(sketch_path)
        sketch = cv2.GaussianBlur(sketch, (3, 3), None)
        sketch = cv2.erode(sketch, kernel=(3, 3), iterations=3)

        return image, {'image': sketch}


if __name__ == "__main__":
    dataset = SketchCelebADataset(data_dir='data')
    print(len(dataset))
    image, cond = dataset[0]
    sketch = cond['image']
    print(image.shape, sketch.shape)

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Target Image')

    plt.subplot(1, 2, 2)
    plt.imshow(sketch, cmap='gray')
    plt.title('Sketch Image')

    plt.show()