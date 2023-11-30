import time
import torch
import hydra
import pyrootutils
from omegaconf import DictConfig
from torchvision.utils import save_image

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.diffusion import ConditionDiffusionModule
from src.models.diffusion.sampler import DDPMSampler


@hydra.main(version_base=None,
            config_path="../configs",
            config_name="inference")
def my_app(cfg: DictConfig):
    print(cfg.ckpt_path)
    model = ConditionDiffusionModule.load_from_checkpoint(cfg.ckpt_path)
    model.net.sampler = DDPMSampler(beta_schedule='linear')

    model.eval()
    model.to(cfg.device)
    cond = torch.tensor([i for i in range(0, 10)] * 10, dtype=torch.int64)

    n_samples = cfg.grid_shape[0] * cfg.grid_shape[1]
    with model.ema_scope():
        samples = model.net.sample(num_sample=n_samples,
                                   device=cfg.device,
                                   cond=cond.to(cfg.device),
                                   prog_bar=True,
                                   repeat_noise=False)

    images = samples[-1].cpu()
    images = (images * cfg.std + cfg.mean).clamp(0, 1)
    filename = '/'.join(cfg.ckpt_path.split('/')[:-1])

    print(filename)
    save_image(images, filename + '/gen_image.jpg', nrow=cfg.grid_shape[0])


if __name__ == "__main__":
    start_time = time.time()
    my_app()
    end_time = time.time()
    print('total time: ', (end_time - start_time) / 60)
