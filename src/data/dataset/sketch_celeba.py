import glob
import imageio
import os.path as osp
from torch.utils.data import Dataset
import cv2

class SketchCelebADataset(Dataset):
    dataset_dir = 'sketch_celeba'
    dataset_url = 'https://drive.google.com/drive/folders/1TxsSzPhZsJNijIXPINv05IUWhG3vBU-X?fbclid=IwAR3o4qVWRF16Q1FJRivhYsXaUZf7nFcobqz04d4-TsR3dPWtXqjL_Gl_wCg'

    def __init__(self,
                 data_dir: str = 'data') -> None:
        """
            data_dir:
        """
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.img_paths = glob.glob(f"{self.dataset_dir}/images/*.jpg")
    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        name = img_path.split('/')
        sketch_path = osp.join(self.dataset_dir, 'sketch', name[-1])

        image = imageio.v2.imread(img_path)
        sketch = imageio.v2.imread(sketch_path)
        sketch = cv2.GaussianBlur(sketch, (3, 3), None)
        sketch = cv2.erode(sketch, kernel=(3, 3), iterations=3)
        return image, sketch
    
if __name__ == "__main__":
    dataset = SketchCelebADataset(data_dir='data')
    print(len(dataset))
    image, sketch = dataset[0]
    print(image.shape, sketch.shape)
    
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Target Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(sketch, cmap='gray')
    plt.title('Sketch Image')

    plt.show()