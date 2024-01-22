from typing import Tuple
from PIL import Image, ImageDraw
import random
import numpy as np
from torch.utils.data import Dataset


class DummyDataset(Dataset):

    dataset_dir = 'dummy'

    def __init__(
        self,
        data_dir: str = 'data',
        num_shapes: int = 4,
        size: int = 256,
        range_shape_size: Tuple[int, int] = (30, 60)) -> None:
        super().__init__()
        self.num_shapes = num_shapes
        self.size = size
        self.range_shape_size = range_shape_size

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        image, mask = self.generate_shape_image()
        image = np.array(image)
        mask = np.array(mask)
        return image, {'image': mask}

    def generate_shape_image(self):
        # Create a black image
        image = Image.new('RGB', (self.size, self.size), color=(0, 0, 0))
        mask = Image.new('L', (self.size, self.size), color=0)

        draw = ImageDraw.Draw(image)
        mask_draw = ImageDraw.Draw(mask)

        for _ in range(self.num_shapes):
            shape_type = random.choice(
                ['circle', 'square', 'triangle', 'ellipse'])
            shape_color = (random.randint(0, 255), random.randint(0, 255),
                           random.randint(0, 255))
            shape_size = random.randint(self.range_shape_size[0],
                                        self.range_shape_size[1])

            # Random position
            x = random.randint(0, self.size - shape_size)
            y = random.randint(0, self.size - shape_size)

            if shape_type == 'circle':
                draw.ellipse([x, y, x + shape_size, y + shape_size],
                             fill=shape_color)
                mask_draw.ellipse([x, y, x + shape_size, y + shape_size],
                                  fill=255)
            elif shape_type == 'square':
                draw.rectangle([x, y, x + shape_size, y + shape_size],
                               fill=shape_color)
                mask_draw.rectangle([x, y, x + shape_size, y + shape_size],
                                    fill=255)
            elif shape_type == 'triangle':
                triangle_points = [(x, y), (x + shape_size, y),
                                   (x + shape_size / 2, y + shape_size)]
                draw.polygon(triangle_points, fill=shape_color)
                mask_draw.polygon(triangle_points, fill=255)
            elif shape_type == 'ellipse':
                draw.ellipse([x, y, x + shape_size, y + shape_size // 2],
                             fill=shape_color)
                mask_draw.ellipse([x, y, x + shape_size, y + shape_size // 2],
                                  fill=255)
        return image, mask


if __name__ == "__main__":
    dataset = DummyDataset(data_dir='data')
    print(len(dataset))
    image, cond = dataset[0]
    mask = cond['image']
    print(image.shape, mask.shape)

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title('Mask')
    plt.show()