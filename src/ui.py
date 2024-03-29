import os.path as osp
import gradio as gr

import torch
import numpy as np
from torchvision import transforms as T

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.diffusion import ConditionDiffusionModule
from src.models.diffusion.sampler import DDPMSampler, DDIMSampler

checkpoint = '/data/hpc/huynhspm/logs/train_segmentation_diffusion/runs/2024-03-21_18-21-27/checkpoints/last.ckpt'
device = torch.device("cuda:1")

model = ConditionDiffusionModule.load_from_checkpoint(checkpoint)
model = model.eval().to(device)

mean, std = 0.5, 0.5
w, h = 256, 256
transform = T.Compose(
    [T.ToTensor(),
     T.Resize((w, h), antialias=True),
     T.Normalize(mean, std)])


def post_process(image: torch.Tensor):
    image = image.moveaxis(-3, -1)
    image = torch.cat([image, image, image], dim=-1)
    return image.cpu().detach().numpy()


def inference(sampler: str, image: np.array, target: np.array):
    image = transform(image).unsqueeze(0).to(device)
    print('Image shape:', image.shape)

    cond = {'image': image}

    if sampler == 'ddim':
        model.net.sampler = DDIMSampler(beta_schedule='linear')
    elif sampler == 'ddpm':
        model.net.sampler = DDPMSampler(beta_schedule='linear')
    else:
        raise NotImplementedError(f"unknown search: {sampler}")

    model.to(device)

    n_ensemble = 5
    masks = []
    for _ in range(n_ensemble):
        samples = model.net.sample(num_sample=1, device=device, cond=cond)
        # [b, c, w, h]
        masks.append(samples[-1])

    #  # (n, b, c, w, h) -> (n, c, w, h)
    masks = torch.stack(masks, dim=0).squeeze(dim=1)

    # convert (-1, 1) to (0, 1)
    masks = (masks * std + mean).clamp(0, 1)

    variance = torch.where(masks > 0.5, 1., 0.).var(dim=0)

    ensemble = masks.mean(dim=0)
    ensemble = torch.where(ensemble > 0.5, 1., 0.)

    # (c, w, h) -> (w, h, c)
    masks = post_process(masks)
    ensemble = post_process(ensemble)
    variance = post_process(variance)

    return masks[0], masks[1], masks[2], masks[3], masks[4], ensemble, variance


demo = gr.Interface(
    inference,
    inputs=[
        gr.Radio(["ddpm", "ddim"], label="sampler", value="ddim"),
        gr.Image(type="numpy", label="Condition", height=h, width=w),
        gr.Image(type="numpy", label="Target", height=h, width=w),
    ],
    outputs=[
        gr.Image(type="numpy", label="Mask 0", height=h, width=w),
        gr.Image(type="numpy", label="Mask 1", height=h, width=w),
        gr.Image(type="numpy", label="Mask 2", height=h, width=w),
        gr.Image(type="numpy", label="Mask 3", height=h, width=w),
        gr.Image(type="numpy", label="Mask 4", height=h, width=w),
        gr.Image(type="numpy", label="# Ensemble", height=h, width=w),
        gr.Image(type="numpy", label="Confidence map", height=h, width=w),
    ],
    examples=[
        ["ddim"],
        ["ddpm"],
    ],
)

if __name__ == "__main__":
    demo.launch()