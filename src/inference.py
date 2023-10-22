from typing import List, Optional
import os
import time
import torch
import hydra
import imageio
import numpy as np
import pyrootutils
from omegaconf import DictConfig
from torchvision.utils import make_grid, save_image
from albumentations import Compose

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.diffusion import ConditionDiffusionModule

def get_sketch(img_paths: List[str], transform: Optional[Compose] = None):
    imgs = []
    for img_path in img_paths:
        img = imageio.v2.imread(img_path)
        img = transform(image=np.array(img))["image"]
        imgs.append(img)
    imgs = torch.stack(imgs, dim=0)
    print(imgs.shape)

@hydra.main(version_base=None,
            config_path="../configs",
            config_name="inference")
def my_app(cfg: DictConfig):
    checkpoint = os.path.join(cfg.checkpoint_dir, cfg.checkpoint + ".ckpt")

    print(checkpoint)
    model = ConditionDiffusionModule.load_from_checkpoint(checkpoint)

    model.eval()
    model.to(cfg.device)

    mean = torch.Tensor(cfg.mean)
    std = torch.Tensor(cfg.std)
    transform = cfg.get('data/transform_val')
    cond = get_sketch(cfg.img_paths, transform=hydra.utils.instantiate(transform))
    
    # with model.ema_scope():
    #     batch_samples = model.net.get_p_sample(
    #             num_sample=cond.shape[0],
    #             gen_type=cfg.gen_type,
    #             device=cfg.device,
    #             cond=cond,
    #             prog_bar=True)
        
    # images = batch_samples[-1].cpu()
    # images = (images * std + mean).clamp(0, 1)
    # filename = cfg.checkpoint_dir + f"{cfg.checkpoint}_{cfg.gen_type}"
    # save_image(images, filename + '.jpg', nrow=cfg.gen_shape[0])

if __name__ == "__main__" :
    start_time = time.time()
    my_app()
    end_time = time.time()
    print('total time: ', (end_time - start_time) / 60)
