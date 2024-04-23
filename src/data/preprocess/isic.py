from typing import List, Tuple
import cv2
import math
import glob
import imageio
import numpy as np
import os.path as osp
from pickle import dump, load
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from tqdm import tqdm
import sys

sys.setrecursionlimit(10000)


class Boundary:

    directions: List[Tuple[int, int]] = [[-1, 0], [-1, 1], [0, 1], [1, 1],
                                         [1, 0], [1, -1], [0, -1], [-1, -1]]
    mask_path: str = None
    full_points: np.array = None
    discrete_points: np.array = None
    centroid: np.array = None  # translation
    scale_matrix: np.array = None
    rotation_matrix: np.array = None
    vec: np.array = None

    def __init__(self,
                 mask_path: str = None,
                 mask: np.array = None,
                 vec: np.array = None) -> None:
        assert not (mask_path is None and mask is None and vec is None)
        if vec is None:
            self.mask_path = mask_path
            self.prepare(mask)
            self.discretized_boundary()
        else:
            pass

    def prepare(self, mask: np.array = None):
        if mask is None:
            mask = imageio.v2.imread(self.mask_path)
        distTransform = cv2.distanceTransform(mask, cv2.DIST_L1, 3)

        boundary_image = (distTransform == 1)

        point_y, point_x = np.where(boundary_image)
        points = np.stack((point_x, point_y), axis=1)

        # for invariant translation
        self.centroid = points.mean(axis=0)
        distance = np.sum((points - self.centroid)**2, axis=1)
        id_max_distance = np.argmax(distance)

        start = np.array([point_x[id_max_distance], point_y[id_max_distance]])
        self.full_points = []
        self.flood_fill(boundary_image, start, self.full_points)
        self.full_points = np.stack(self.full_points, axis=0)

        new_distance = np.sum((self.full_points - self.centroid)**2, axis=1)
        new_distance[0] = 0
        id_se_max_distance = np.argmax(new_distance)
        if id_se_max_distance * 2 < new_distance.shape[0]:
            self.full_points[1:] = np.flip(self.full_points[1:], axis=0)

        # for invariant scale
        scale = np.sqrt(2 / distance[id_max_distance])
        print('Max distance:', scale)
        self.scale_matrix = np.array([[scale, 0], [0, scale]])

        # for invariant rotation
        delta = self.full_points[0] - self.centroid
        slope = delta[1] / delta[0]
        angle = -np.arctan(slope)
        print('Angle:', angle, angle / np.pi * 180)

        cos = np.cos(angle)
        sin = np.sin(angle)

        print('Cos, Sin:', cos, sin)

        self.rotation_matrix = np.array([[cos, -sin], [sin, cos]])

    def discretized_boundary(self, n_point: int = 500):
        idx = np.linspace(0,
                          len(self.full_points) - 1, n_point).astype(np.int64)
        self.discrete_points = self.full_points[idx].astype(np.float64)

    def invariant_translation(self):
        self.discrete_points -= self.centroid

    def invariant_rotation(self):
        self.discrete_points = [
            self.rotation_matrix @ discrete_point[..., np.newaxis]
            for discrete_point in self.discrete_points
        ]
        self.discrete_points = np.stack(self.discrete_points,
                                        axis=0).squeeze(-1)
        print('Max point', self.discrete_points[0])

    def invariant_scale(self):
        self.discrete_points = [
            self.scale_matrix @ discrete_point[..., np.newaxis]
            for discrete_point in self.discrete_points
        ]
        self.discrete_points = np.stack(self.discrete_points,
                                        axis=0).squeeze(-1)

    def boundaryCoord_2_imageCoord(self):
        pass

    def imageCoord_2_boundaryCoord(self):
        self.invariant_translation()
        self.invariant_rotation()
        # self.invariant_scale()

    def discrete_boundary_2_vec(self):
        self.imageCoord_2_boundaryCoord()
        self.vec = self.discrete_points.flatten()

    def vec_2_discrete_boundary(self):
        pass

    def flood_fill(self, boundary: np.array, cur_point: np.array,
                   points: List[np.array]):

        boundary[cur_point[1], cur_point[0]] = False
        points.append(cur_point)

        for direction in self.directions:
            next_point = cur_point + np.array(direction)

            if next_point[1] < 0 or next_point[1] >= boundary.shape[
                    0] or next_point[0] < 0 or next_point[0] >= boundary.shape[
                        1] or not boundary[next_point[1], next_point[0]]:
                continue

            boundary[next_point[1], next_point[0]] = False
            points.append(next_point)

            for direct in self.directions:
                next_next_point = next_point + np.array(direct)

                if next_next_point[1] < 0 or next_next_point[
                        1] >= boundary.shape[0] or next_next_point[
                            0] < 0 or next_next_point[0] >= boundary.shape[
                                1] or not boundary[next_next_point[1],
                                                   next_next_point[0]]:
                    continue

                self.flood_fill(boundary, next_next_point, points)


class ISICDataset(Dataset):

    dataset_dir = 'isic'
    dataset_url = 'https://challenge.isic-archive.com/data/'
    boundaries: List[Boundary] = []

    def __init__(self, data_dir: str = 'data') -> None:
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        img_dir = [
            f"{self.dataset_dir}/ISBI2016_ISIC_Part3B_Test_Data/*.jpg",
            f"{self.dataset_dir}/ISBI2016_ISIC_Part3B_Training_Data/*.jpg"
        ]

        self.img_paths = glob.glob(img_dir[0]) + glob.glob(img_dir[1])
        self.prepare_data()

    def prepare_data(self, boundaries_path: str = "boundaries.pkl") -> None:

        boundaries_path = osp.join(self.dataset_dir, boundaries_path)

        if osp.exists(boundaries_path):
            with open(boundaries_path, "rb") as file:
                self.boundaries = load(file)
            return

        self.boundaries = []
        self.full_boundaries = []

        for img_path in tqdm(self.img_paths):
            boundary = Boundary(mask_path=img_path[0:-4] + '_Segmentation.png')
            self.boundaries.append(boundary)

        with open(boundaries_path, "wb") as file:
            dump(self.boundaries, file)

    def __len__(self):
        return len(self.boundaries)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = img_path[0:-4] + '_Segmentation.png'
        image = imageio.v2.imread(img_path)
        mask = imageio.v2.imread(mask_path)

        assert self.boundaries[index] is not None

        return image, {'image': mask}, self.boundaries[index]


def get_max_distance(mask_path: str):
    mask = imageio.v2.imread(mask_path)
    distTransform = cv2.distanceTransform(mask, cv2.DIST_L1, 3)

    boundary_image = (distTransform == 1)

    point_y, point_x = np.where(boundary_image)
    points = np.stack((point_x, point_y), axis=1)

    print(points.shape)

    distance = np.sum((points)**2, axis=1)
    id_max_distance = np.argmax(distance)

    print(distance[id_max_distance])

    max_distance = np.sqrt(distance[id_max_distance] / 2)
    print(max_distance)
    print('-' * 100)


def test_scale(mask_path: str):
    mask = imageio.v2.imread(mask_path)
    resize_mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))

    plt.subplot(1, 2, 1)
    plt.imshow(mask)
    plt.subplot(1, 2, 2)
    plt.imshow(resize_mask)
    plt.show()

    boundary = Boundary(mask=mask)
    resize_boundary = Boundary(mask=resize_mask)

    boundary.discrete_boundary_2_vec()
    resize_boundary.discrete_boundary_2_vec()

    print(np.allclose(boundary.vec, resize_boundary.vec))


def test_translate(mask_path: str):
    print('-' * 10, "Test translate", '-' * 10)

    mask = imageio.v2.imread(mask_path)
    translate_mask = np.concatenate(
        (mask[20:], np.zeros((20, mask.shape[1]), dtype=mask.dtype)), axis=0)

    boundary = Boundary(mask=mask)
    translate_boundary = Boundary(mask=translate_mask)

    for point in boundary.discrete_points[::5]:
        cv2.circle(mask, point.astype(np.int64), 1, (0, 255, 0),
                   int(mask.shape[0] / 100))

    for point in translate_boundary.discrete_points[::5]:
        cv2.circle(translate_mask, point.astype(np.int64), 1, (0, 255, 0),
                   int(mask.shape[0] / 100))

    plt.subplot(1, 2, 1)
    plt.imshow(mask)
    plt.subplot(1, 2, 2)
    plt.imshow(translate_mask)
    plt.show()

    boundary.discrete_boundary_2_vec()
    translate_boundary.discrete_boundary_2_vec()

    print(np.allclose(boundary.vec, translate_boundary.vec))


def test_rotate(mask_path: str):
    print('-' * 10, "Test rotate", '-' * 10)

    mask = imageio.v2.imread(mask_path)
    mask = cv2.resize(mask, (500, 500))

    rotate_matrix = cv2.getRotationMatrix2D(
        (mask.shape[1] / 2, mask.shape[0] / 2), 30, 1.0)
    rotate_mask = cv2.warpAffine(mask, rotate_matrix,
                                 (mask.shape[1], mask.shape[0]))

    boundary = Boundary(mask=mask)
    rotate_boundary = Boundary(mask=rotate_mask)

    print(rotate_mask.shape)

    # for point in boundary.discrete_points[::5]:
    #     cv2.circle(mask, point.astype(np.int64), 1, (0, 255, 0),
    #                int(mask.shape[0] / 100))
    # print(boundary.centroid)
    cv2.circle(mask, boundary.centroid.astype(np.int64), 3, (0, 255, 0), 3)
    cv2.circle(mask, boundary.discrete_points[0].astype(np.int64), 5,
               (0, 255, 0), 3)

    # for point in rotate_boundary.discrete_points[::5]:
    #     cv2.circle(rotate_mask, point.astype(np.int64), 1, (0, 255, 0),
    #                int(mask.shape[0] / 100))
    cv2.circle(rotate_mask, rotate_boundary.centroid.astype(np.int64), 3,
               (0, 255, 0), 3)
    cv2.circle(rotate_mask,
               rotate_boundary.discrete_points[0].astype(np.int64), 5,
               (0, 255, 0), 3)

    plt.subplot(1, 2, 1)
    plt.imshow(mask)
    plt.subplot(1, 2, 2)
    plt.imshow(rotate_mask)
    plt.show()

    boundary.discrete_boundary_2_vec()
    rotate_boundary.discrete_boundary_2_vec()

    print(boundary.vec[0:5])
    print(rotate_boundary.vec[0:5])
    i = -1
    print(np.allclose(boundary.vec[:i], rotate_boundary.vec[:i], atol=200))
    print(np.max(np.abs(boundary.vec - rotate_boundary.vec)))


if __name__ == "__main__":

    # test_translate(
    #     mask_path=
    #     "data/isic/ISBI2016_ISIC_Part3B_Test_Data/ISIC_0009918_Segmentation.png"
    # )

    # test_rotate(
    #     mask_path=
    #     "data/isic/ISBI2016_ISIC_Part3B_Test_Data/ISIC_0009958_Segmentation.png"
    # )

    # test_scale(
    #     mask_path=
    #     "data/isic/ISBI2016_ISIC_Part3B_Test_Data/ISIC_0009918_Segmentation.png"
    # )

    print('-' * 100)
    dataset = ISICDataset(data_dir='data')
    print(len(dataset))

    id = np.random.randint(0, len(dataset))
    print('ID ID ID:', id)

    image, cond, boundary = dataset[id]
    mask = cond['image']

    print(boundary.mask_path)

    # resize_mask = img = cv2.resize(mask, (512, 384))
    # resize_boundary = Boundary(mask=resize_mask)

    # resize_boundary.imageCoord_2_frameCoord()
    # boundary.imageCoord_2_frameCoord()

    # i = 0
    # print()
    # print(boundary.discrete_points[i].flatten(),
    #       resize_boundary.discrete_points[i].flatten())

    print(image.shape, mask.shape)
    print(np.unique(mask))

    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    new_mask = mask.copy()
    new_image = image.copy()

    for point in boundary.full_points[::5]:
        cv2.circle(new_image, point.astype(np.int64), 1, (0, 255, 0),
                   int(image.shape[0] / 100))
        cv2.circle(new_mask, point.astype(np.int64), 1, (255, 0, 0),
                   int(image.shape[0] / 100))

    for point in boundary.discrete_points[::5]:
        cv2.circle(image, point.astype(np.int64), 1, (0, 255, 0),
                   int(image.shape[0] / 100))
        cv2.circle(mask, point.astype(np.int64), 1, (255, 0, 0),
                   int(image.shape[0] / 100))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('Discrete Points Image')

    plt.subplot(2, 2, 2)
    plt.imshow(new_image)
    plt.title('Full Points Image')

    plt.subplot(2, 2, 3)
    plt.imshow(mask)
    plt.title('Discrete Points Mask')

    plt.subplot(2, 2, 4)
    plt.imshow(new_mask)
    plt.title('Full Points Mask')
    plt.show()
