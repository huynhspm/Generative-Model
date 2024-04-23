from typing import List
import os
import cv2
import glob
import shutil
import random
import pydicom
import numpy as np
import pylidc as pl
from tqdm import tqdm
from pathlib import Path
from pylidc.utils import consensus

from sklearn.cluster import KMeans
from skimage import measure, morphology
from skimage.measure import regionprops
from scipy.ndimage import median_filter

# from medpy.filter.smoothing import anisotropic_diffusion


class MakeDataSet:

    def __init__(self,
                 patient_dirs: List[str],
                 data_dir: str,
                 mask_threshold: float,
                 padding: int,
                 confidence_level: float = 0.5):
        self.patient_dirs = patient_dirs
        self.multi_annotations_dir = os.path.join(data_dir,
                                                  "Multi-Annotations")
        self.one_annotation_dir = os.path.join(data_dir, "One-Annotation")
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding, padding), (padding, padding), (0, 0)]

    def prepare_dataset(self):

        # Make directory
        Path(self.multi_annotations_dir).mkdir(parents=True, exist_ok=True)
        Path(self.one_annotation_dir).mkdir(parents=True, exist_ok=True)

        multi_annotations_image_dir = os.path.join(self.multi_annotations_dir,
                                                   "Image")
        multi_annotations_mask_dir = os.path.join(self.multi_annotations_dir,
                                                  "Mask")
        one_annotation_image_dir = os.path.join(self.one_annotation_dir,
                                                "Image")
        one_annotation_mask_dir = os.path.join(self.one_annotation_dir, "Mask")

        # Make directory
        Path(multi_annotations_image_dir).mkdir(parents=True, exist_ok=True)
        Path(multi_annotations_mask_dir).mkdir(parents=True, exist_ok=True)
        Path(one_annotation_image_dir).mkdir(parents=True, exist_ok=True)
        Path(one_annotation_mask_dir).mkdir(parents=True, exist_ok=True)

        for patient_dir in tqdm(self.patient_dirs):
            pid = patient_dir  #LIDC-IDRI-0001~

            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            dicom_path = scan.get_path_to_dicom_files()

            files = [
                file for file in os.listdir(dicom_path)
                if file.endswith('.dcm')
            ]

            print(
                "Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}"
                .format(pid, vol.shape, len(nodules_annotation)))

            if len(nodules_annotation) > 0:
                nodule_index = []
                mask_list = []
                masks_list = []
                one_annotation_slice = []

                for _, nodule in enumerate(nodules_annotation):
                    # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                    # This current for loop iterates over total number of nodules in a single patient
                    mask, cbbox, masks = consensus(nodule, self.c_level,
                                                   self.padding)
                    slices = list(range(cbbox[2].start, cbbox[2].stop))

                    masks = np.stack(masks, axis=0)
                    for nodule_slice in range(mask.shape[2]):
                        # This second for loop iterates over each single nodule.
                        # There are some mask sizes that are too small. These may hinder training.
                        if np.sum(mask[:, :,
                                       nodule_slice]) <= self.mask_threshold:
                            continue

                        if masks.shape[
                                0] < 4 and slices[nodule_slice] not in one_annotation_slice:
                            # one slice can have more nodules.
                            # one nodule in slice has not enough 4 annotation -> this slice is in one annotation group
                            one_annotation_slice.append(slices[nodule_slice])
                        else:
                            # only get 4 annotations
                            masks = masks[:4]

                        if slices[nodule_slice] not in nodule_index:
                            nodule_index.append(slices[nodule_slice])
                            mask_list.append(mask[:, :, nodule_slice])
                            masks_list.append(masks[:, :, :, nodule_slice])
                        else:
                            # one slice can have more nodules -> merge all nodules that it is in same slice
                            index = nodule_index.index(slices[nodule_slice])
                            mask_list[index] = np.logical_or(
                                mask_list[index], mask[:, :, nodule_slice])

                            # logical_or must have same number of annotations
                            if slices[
                                    nodule_slice] not in one_annotation_slice:
                                masks_list[index] = np.logical_or(
                                    masks_list[index], masks[:, :, :,
                                                             nodule_slice])

                for slice in range(vol.shape[2]):
                    # slice have nodule
                    if slice in nodule_index:

                        index = nodule_index.index(slice)
                        image_path = os.path.join(dicom_path, files[slice])

                        ds = pydicom.dcmread(image_path)
                        intercept = ds.RescaleIntercept
                        slope = ds.RescaleSlope

                        image = vol[:, :, slice]
                        # lung_segmented = segment_lung(image)
                        mask = mask_list[index]

                        # preprocess
                        image = ct_normalize(image, slope, intercept)
                        image = resize_image(image)
                        # lung_segmented = resize_image(lung_segmented)
                        mask = resize_mask(mask)

                        if slice in one_annotation_slice:
                            patient_image_dir = os.path.join(
                                one_annotation_image_dir, pid)
                            patient_mask_dir = os.path.join(
                                one_annotation_mask_dir, pid)

                            # Make directory
                            Path(patient_image_dir).mkdir(parents=True,
                                                          exist_ok=True)
                            Path(patient_mask_dir).mkdir(parents=True,
                                                         exist_ok=True)

                            np.save(f"{patient_image_dir}/slice_{slice}",
                                    image)
                            # np.save(patient_image_dir / f"slice_{slice}_lung", lung_segmented)
                            np.save(f"{patient_mask_dir}/slice_{slice}", mask)
                        else:
                            masks = masks_list[index]

                            # preprocess
                            mask_0 = resize_mask(masks[0, ...])
                            mask_1 = resize_mask(masks[1, ...])
                            mask_2 = resize_mask(masks[2, ...])
                            mask_3 = resize_mask(masks[3, ...])

                            # Make directory
                            patient_image_dir = os.path.join(
                                multi_annotations_image_dir, pid)
                            patient_mask_dir = os.path.join(
                                multi_annotations_mask_dir, pid)

                            # Make directory
                            Path(patient_image_dir).mkdir(parents=True,
                                                          exist_ok=True)
                            Path(patient_mask_dir).mkdir(parents=True,
                                                         exist_ok=True)

                            np.save(f"{patient_image_dir}/slice_{slice}",
                                    image)
                            # np.save(image_dir / f"slice_{slice}_lung", lung_segmented)
                            np.save(f"{patient_mask_dir}/slice_{slice}_e",
                                    mask)
                            np.save(f"{patient_mask_dir}/slice_{slice}_0",
                                    mask_0)
                            np.save(f"{patient_mask_dir}/slice_{slice}_1",
                                    mask_1)
                            np.save(f"{patient_mask_dir}/slice_{slice}_2",
                                    mask_2)
                            np.save(f"{patient_mask_dir}/slice_{slice}_3",
                                    mask_3)

                    # slice no nodule
                    else:
                        pass


def resize_image(image, size=128):
    resized_image = cv2.resize(image, (size, size),
                               interpolation=cv2.INTER_AREA)
    return resized_image


def resize_mask(mask, size=128):
    resized_mask = cv2.resize(mask.astype(float), (size, size))
    resized_mask = resized_mask.astype(bool)
    return resized_mask


def ct_normalize(image, slope, intercept):
    image[image == -0] = 0
    image = image * slope + intercept
    image[image > 400] = 400
    image[image < -1000] = -1000
    return image


# custom anisotropic_diffusion, error with lib
def anisotropic_diffusion(img,
                          niter=1,
                          kappa=50,
                          gamma=0.1,
                          voxelspacing=None,
                          option=1):
    r"""
    Edge-preserving, XD Anisotropic diffusion.


    Parameters
    ----------
    img : array_like
        Input image (will be cast to np.float).
    niter : integer
        Number of iterations.
    kappa : integer
        Conduction coefficient, e.g. 20-100. ``kappa`` controls conduction
        as a function of the gradient. If ``kappa`` is low small intensity
        gradients are able to block conduction and hence diffusion across
        steep edges. A large value reduces the influence of intensity gradients
        on conduction.
    gamma : float
        Controls the speed of diffusion. Pick a value :math:`<= .25` for stability.
    voxelspacing : tuple of floats or array_like
        The distance between adjacent pixels in all img.ndim directions
    option : {1, 2, 3}
        Whether to use the Perona Malik diffusion equation No. 1 or No. 2,
        or Tukey's biweight function.
        Equation 1 favours high contrast edges over low contrast ones, while
        equation 2 favours wide regions over smaller ones. See [1]_ for details.
        Equation 3 preserves sharper boundaries than previous formulations and
        improves the automatic stopping of the diffusion. See [2]_ for details.

    Returns
    -------
    anisotropic_diffusion : ndarray
        Diffused image.

    Notes
    -----
    Original MATLAB code by Peter Kovesi,
    School of Computer Science & Software Engineering,
    The University of Western Australia,
    pk @ csse uwa edu au,
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal,
    Department of Pharmacology,
    University of Oxford,
    <alistair.muldal@pharm.ox.ac.uk>

    Adapted to arbitrary dimensionality and added to the MedPy library Oskar Maier,
    Institute for Medical Informatics,
    Universitaet Luebeck,
    <oskar.maier@googlemail.com>

    June 2000  original version. -
    March 2002 corrected diffusion eqn No 2. -
    July 2012 translated to Python -
    August 2013 incorporated into MedPy, arbitrary dimensionality -

    References
    ----------
    .. [1] P. Perona and J. Malik.
       Scale-space and edge detection using ansotropic diffusion.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       12(7):629-639, July 1990.
    .. [2] M.J. Black, G. Sapiro, D. Marimont, D. Heeger
       Robust anisotropic diffusion.
       IEEE Transactions on Image Processing,
       7(3):421-432, March 1998.
    """
    # define conduction gradients functions
    if option == 1:

        def condgradient(delta, spacing):
            return np.exp(-(delta / kappa)**2.) / float(spacing)
    elif option == 2:

        def condgradient(delta, spacing):
            return 1. / (1. + (delta / kappa)**2.) / float(spacing)
    elif option == 3:
        kappa_s = kappa * (2**0.5)

        def condgradient(delta, spacing):
            top = 0.5 * ((1. - (delta / kappa_s)**2.)**2.) / float(spacing)
            return np.where(np.abs(delta) <= kappa_s, top, 0)

    # initialize output array
    out = np.array(img, dtype=np.float32, copy=True)

    # set default voxel spacing if not supplied
    if voxelspacing is None:
        voxelspacing = tuple([1.] * img.ndim)

    # initialize some internal variables
    deltas = [np.zeros_like(out) for _ in range(out.ndim)]

    for _ in range(niter):

        # calculate the diffs
        for i in range(out.ndim):
            slicer = [
                slice(None, -1) if j == i else slice(None)
                for j in range(out.ndim)
            ]
            deltas[i][tuple(slicer)] = np.diff(out, axis=i)

        # update matrices
        matrices = [
            condgradient(delta, spacing) * delta
            for delta, spacing in zip(deltas, voxelspacing)
        ]

        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't as questions. just do it. trust me.
        for i in range(out.ndim):
            slicer = [
                slice(1, None) if j == i else slice(None)
                for j in range(out.ndim)
            ]
            matrices[i][tuple(slicer)] = np.diff(matrices[i], axis=i)

        # update the image
        out += gamma * (np.sum(matrices, axis=0))

    return out


def segment_lung(img):
    #function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    """
    This segments the Lung Image(Don't get confused with lung nodule segmentation)
    """
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std

    middle = img[100:400, 100:400]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    #remove the underflow bins
    img[img == max] = mean
    img[img == min] = mean

    #apply median filter
    img = median_filter(img, size=3)
    #apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    img = anisotropic_diffusion(img)

    kmeans = KMeans(n_clusters=2).fit(
        np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))
    labels = measure.label(dilation)
    # label_vals = np.unique(labels)
    regions = regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    mask = np.ndarray([512, 512], dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
    # mask consists of 1 and 0. Thus by mutliplying with the orginial image, sections with 1 will remain
    return mask * img


def visualize(data_dir: str, pid: str = 'LIDC-IDRI-0001', slice: int = 86):
    image_path = os.path.join(data_dir, f"Image/{pid}/slice_{slice}.npy")
    # mask_lung_path = os.path.join(data_dir, f"mask/{pid}/slice_{slice}.npy")
    mask_e_path = os.path.join(data_dir, f"Mask/{pid}/slice_{slice}_e.npy")
    mask_0_path = os.path.join(data_dir, f"Mask/{pid}/slice_{slice}_0.npy")
    mask_1_path = os.path.join(data_dir, f"Mask/{pid}/slice_{slice}_1.npy")
    mask_2_path = os.path.join(data_dir, f"Mask/{pid}/slice_{slice}_2.npy")
    mask_3_path = os.path.join(data_dir, f"Mask/{pid}/slice_{slice}_3.npy")

    image = np.load(image_path)
    print(image.max(), image.min())

    # mask_lung = np.load(mask_lung_path)
    mask_e = np.load(mask_e_path)
    mask_0 = np.load(mask_0_path)
    mask_1 = np.load(mask_1_path)
    mask_2 = np.load(mask_2_path)
    mask_3 = np.load(mask_3_path)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 12))

    row = 2
    col = 4

    plt.subplot(row, col, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image')

    plt.subplot(row, col, 2)
    plt.imshow(mask_e, cmap='gray')
    plt.title('Mask-ensemble')

    plt.subplot(row, col, 3)
    plt.imshow(mask_0, cmap='gray')
    plt.title('Mask-0')

    plt.subplot(row, col, 4)
    plt.imshow(mask_1, cmap='gray')
    plt.title('Mask-1')

    plt.subplot(row, col, 5)
    plt.imshow(mask_2, cmap='gray')
    plt.title('Mask-2')

    plt.subplot(row, col, 6)
    plt.imshow(mask_3, cmap='gray')
    plt.title('Mask-3')

    # plt.subplot(row, col, 7)
    # plt.imshow(mask_lung, cmap='gray')
    # plt.title('Mask-Lung')

    plt.savefig("image.png")


def split_data(src_dir: str, des_dir):
    n_data = len(glob.glob(os.path.join(src_dir, "Image/*/slice_*.npy")))
    
    patients = os.listdir(os.path.join(src_dir, "Image"))
    
    random.seed(42)
    random.shuffle(patients)

    train_dir = os.path.join(des_dir, "Train")
    val_dir = os.path.join(des_dir, "Val")
    test_dir = os.path.join(des_dir, "Test")
    
    n_val = 0
    n_test = 0
    for patient in patients:
        print(n_val, n_test)
        if n_val * 10 < n_data:
            n_val += len(glob.glob(f"{src_dir}/Image/{patient}/slice_*.npy"))
            shutil.move(f"{src_dir}/Image/{patient}", f"{val_dir}/Image/{patient}")
            shutil.move(f"{src_dir}/Mask/{patient}", f"{val_dir}/Mask/{patient}")
            
        elif n_test * 10 < n_data:
            n_test += len(glob.glob(f"{src_dir}/Image/{patient}/slice_*.npy"))
            shutil.move(f"{src_dir}/Image/{patient}", f"{test_dir}/Image/{patient}")
            shutil.move(f"{src_dir}/Mask/{patient}", f"{test_dir}/Mask/{patient}")
            
        else:
            shutil.move(f"{src_dir}/Image/{patient}", f"{train_dir}/Image/{patient}")
            shutil.move(f"{src_dir}/Mask/{patient}", f"{train_dir}/Mask/{patient}")


if __name__ == '__main__':

    #Get Directory setting
    DICOM_DIR = "lidc/Multi-Annotations"
    DATA_DIR = "./data/lidc"

    #Hyper Parameter setting for prepare dataset function
    mask_threshold = 8

    #Hyper Parameter setting for pylidc
    confidence_level = 0.5
    padding = 512

    # I found out that simply using os.listdir() includes the gitignore file
    patient_dirs = [
        dir for dir in os.listdir(DICOM_DIR) if not dir.startswith('.')
    ]
    patient_dirs.sort()

    # dataset = MakeDataSet(patient_dirs=patient_dirs,
    #                       data_dir=DATA_DIR,
    #                       mask_threshold=mask_threshold,
    #                       padding=padding,
    #                       confidence_level=confidence_level)
    # dataset.prepare_dataset()

    # visualize(data_dir = "data/lidc/Multi-Annotations" , pid='LIDC-IDRI-0001', slice=88)

    split_data(src_dir="data/lidc/Multi-Annotations", des_dir="data/lidc/Multi-Annotations")

