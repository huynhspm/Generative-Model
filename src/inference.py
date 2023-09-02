from typing import List
import os
import time
import torch
import hydra
import pyrootutils
from omegaconf import DictConfig
from torchvision.utils import make_grid, save_image

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.generative_models import DiffusionModule, ConditionDiffusionModule
from src.generative_models.components import DiffusionModel

@hydra.main(version_base=None,
            config_path="../configs",
            config_name="inference")
def my_app(cfg: DictConfig):
    checkpoint = os.path.join(cfg.checkpoint_dir, cfg.checkpoint + ".ckpt")

    print(checkpoint)
    net: DiffusionModel = hydra.utils.instantiate(cfg.get("model")['net'])
    model: DiffusionModule = hydra.utils.instantiate(cfg.get("model"))
    model = model.load_from_checkpoint(checkpoint, net=net)

    model.eval()
    model.to(cfg.device)

    num_samples = cfg.gen_shape[0] * cfg.gen_shape[1]
    mean = torch.Tensor(cfg.mean).reshape(1, -1, 1, 1)
    std = torch.Tensor(cfg.std).reshape(1, -1, 1, 1)
    cond = None
    if isinstance(model, ConditionDiffusionModule):
        cond = torch.arange(0, 10, device=cfg.device).repeat(10)
    
    # Generate samples from denoising process
    if model.use_ema:
        # generate sample by ema_model
        with model.ema_scope():
            batch_samples = model.net.get_p_sample(
                    num_sample=num_samples,
                    gen_type=cfg.gen_type,
                    device=cfg.device,
                    cond=cond,
                    prog_bar=True)
    else:
        batch_samples = model.net.get_p_sample(
                    num_sample=num_samples,
                    gen_type=cfg.gen_type,
                    device=cfg.device,
                    cond=cond,
                    prog_bar=True)

    images = batch_samples[-1].cpu()
    images = (images * std + mean).clamp(0, 1)
    filename = cfg.checkpoint_dir + f"{cfg.checkpoint}_{cfg.gen_type}"
    save_image(images, filename + '.jpg', nrow=cfg.gen_shape[0])

if __name__ == "__main__" :
    start_time = time.time()
    my_app()
    end_time = time.time()
    print('total time: ', (end_time - start_time) / 60)
