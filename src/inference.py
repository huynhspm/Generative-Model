import time
import torch
import random
import imageio
import numpy as np
import pyrootutils
import matplotlib.pyplot as plt
import seaborn as sns 
from tqdm import tqdm

from torchvision import transforms as T
from torchvision.utils import save_image

from torchmetrics.functional import dice
from torchmetrics.functional import jaccard_index as iou

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.diffusion import ConditionDiffusionModule
from src.models.diffusion.sampler import DDPMSampler

mean, std = 0.5, 0.5
w, h = 128, 128
# w, h = 256, 256
transform = T.Compose(
    [T.ToTensor(),
     T.Resize((w, h), antialias=True),
     T.Normalize(mean, std)])

def rescale(image):
    #convert range (-1, 1) to (0, 1)
    return (image * std + mean).clamp(0, 1)


@torch.no_grad()
def infer(dataset: str):
    if dataset == "isic":
        id = "ISIC_0012302"
        image_path = f"data/isic-2018/ISIC2018_Task1-2_Test_Input/{id}.jpg"
        mask_path = f"data/isic-2018/ISIC2018_Task1_Test_GroundTruth/{id}_segmentation.png"
        checkpoint = "logs/train_segmentation_diffusion/runs/2024-05-09_11-25-01/checkpoints/last.ckpt"
    elif dataset == "cvc-clinic":
        id = 100
        image_path = f"data/cvc_clinic/Original/{id}.png"
        mask_path = f"data/cvc_clinic/Ground_Truth/{id}.png"
        checkpoint = "logs/train_segmentation_diffusion/runs/2024-05-08_05-32-03/checkpoints/last.ckpt"
    elif dataset == "lidc":
        patient = "LIDC-IDRI-0141"
        slice = "slice_180"
        image_path = f"data/lidc/Multi-Annotations/Test/Image/{patient}/{slice}.npy"
        mask_path = f"data/lidc/Multi-Annotations/Test/Mask/{patient}/{slice}_e.npy"
        checkpoint = "logs/train_segmentation_diffusion/runs/2024-05-09_12-40-41/checkpoints/last.ckpt"
    elif dataset == "brats":
        patient = "BraTS20_Training_174"
        slice = "105"
        image_path = f"data/brats-2020/Test/{patient}/image_slice_{slice}.npy"
        mask_path = f"data/brats-2020/Test/{patient}/mask_slice_{slice}.npy"
        checkpoint = "logs/train_segmentation_diffusion/runs/2024-05-12_02-03-28/checkpoints/last.ckpt"

    image = np.load(image_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    
    # image = imageio.v2.imread(image_path)
    # mask = imageio.v2.imread(mask_path)

    device = torch.device("cuda:0")
    
    image = transform(image).unsqueeze(0).to(device)
    mask = transform(mask).unsqueeze(0).to(device)

    model = ConditionDiffusionModule.load_from_checkpoint(checkpoint)
    model = model.eval().to(device)

    cond = {"image": image}
    
    n_samples = 5
    samples = []
    for _ in tqdm(range(n_samples)):
        sample = model.net.sample(num_sample=1, cond=cond, device=device)
        
        # n x [b, c, w, h]
        samples.append(sample[-1])

    # b, n, c, w, h
    samples = torch.stack(samples, dim=1)
    
    save_image(samples[0], fp="samples.jpg", nrow=20, padding=3, pad_value=1)
    
    return samples, mask, image

def compute_metrics(samples, target, cond):
    
    target = (rescale(target) > 0.5).to(torch.int64)

    for n_ensemble in range(19):
        
        dice_score = 0
        iou_score = 0
        n_compute = 10
        for _ in range(n_compute):
            ids = random.sample(range(samples.shape[1]), n_ensemble)
            
            ensemble = samples[:, ids, ...].mean(dim=1)
            pred = (rescale(ensemble) > 0.5).to(torch.int64)
        
            dice_score +=  dice(pred, target, threshold=0.5, average='micro', ignore_index=0)
            iou_score += iou(pred, target, threshold=0.5, average='micro', task='binary')
        
        dice_score /= n_compute
        iou_score /= n_compute
        print("{} ensemble: {} {}".format(n_ensemble, dice_score, iou_score))
        print('-' * 20)

if __name__ == "__main__":
    
    dataset = "brats"
    start_time = time.time()
    samples, target, cond = infer(dataset)
    end_time = time.time()
    print('total time: ', (end_time - start_time) / 60)
    
    # compute_metrics(samples, target, cond)

    save_image(rescale(target), fp="target.jpg")

    if dataset == "brats":
        plt.imshow(rescale(cond[0].cpu().moveaxis(0, 2)))
        plt.axis('off')
        plt.savefig('cond.jpg')
    else:
        save_image(rescale(cond), fp="cond.jpg")
    
    ensemble = samples.mean(dim=1) > 0.5
    save_image(ensemble.to(torch.float32), fp="ensemble.jpg")
    
    samples = rescale(samples) > 0.5

    for i in range(samples.shape[1]):
        save_image(samples[:, i, ...].to(torch.float32), fp=f"{i}.jpg")
    
    variance = samples.to(torch.float32).var(dim = 1)
    
    print(variance[0][0].unique())
    sns.heatmap(variance[0][0].cpu()).collections[0].colorbar.ax.tick_params(labelsize=20)
    plt.axis('off')
    plt.savefig("variance.jpg")
    