from typing import List
import os
import shutil
import random
import nibabel
import numpy as np
from pathlib import Path


def preprocess(src_dir: str, des_dir: str, img_types: List[str]):

    # create folder
    Path(des_dir).mkdir(parents=True, exist_ok=True)

    for dir in os.listdir(src_dir):

        patient_src_dir = os.path.join(src_dir, dir)
        patient_des_dir = os.path.join(des_dir, dir)

        if not os.path.isdir(patient_src_dir): continue

        files = os.listdir(patient_src_dir)
        files.sort()

        image = []
        mask = None

        for (file, img_type) in zip(files, img_types):

            if img_type not in file:
                raise IndexError('not match type', file, img_type)

            file_path = os.path.join(patient_src_dir, file)
            img = np.array(nibabel.load(file_path).get_fdata())

            # crop to a size of (224, 224) from (240, 240)
            img = img[8:-8, 8:-8, ...]

            if img_type == 'seg':
                mask = (img > 0).astype(np.uint8) * 255
            else:
                if img.max() <= 0:
                    raise AssertionError('negative value', file, img.max())
                else:
                    img = img / img.max() * 255

                image.append(img)

        image = np.stack(image, axis=2)

        # create folder
        Path(patient_des_dir).mkdir(parents=True, exist_ok=True)

        # exclude the lowest 80 slices and the uppermost 26 slices
        for slice in range(80, 129):
            image_name = f"{patient_des_dir}/image_slice_{slice}"
            mask_name = f"{patient_des_dir}/mask_slice_{slice}"

            print(image_name, mask_name)

            np.save(image_name, image[..., slice])
            if mask is not None:
                np.save(mask_name, mask[..., slice])


def split_data(train_dir, val_dir, n_patient_val=40):
    train_patients = os.listdir(train_dir)

    random.seed(42)
    val_patients = random.sample(train_patients, n_patient_val)

    for val_patient in val_patients:
        shutil.move(f"{train_dir}/{val_patient}", f"{val_dir}/{val_patient}")


if __name__ == "__main__":
    data_dir = "data/brats-2020/"
    data_url = 'https://www.cbica.upenn.edu/MICCAI_BraTS2020_TrainingData'

    train_src_dir = os.path.join(data_dir, 'MICCAI_BraTS2020_TrainingData')
    train_des_dir = os.path.join(data_dir, 'train')
    train_img_types = ['flair', 'seg', 't1', 't1ce', 't2']
    preprocess(src_dir=train_src_dir,
               des_dir=train_des_dir,
               img_types=train_img_types)

    test_src_dir = os.path.join(data_dir, 'MICCAI_BraTS2020_ValidationData')
    test_des_dir = os.path.join(data_dir, 'test')
    test_img_types = ['flair', 't1', 't1ce', 't2']  # no label
    preprocess(src_dir=test_src_dir,
               des_dir=test_des_dir,
               img_types=test_img_types)

    val_dir = os.path.join(data_dir, 'val')
    split_data(train_des_dir, val_dir)
