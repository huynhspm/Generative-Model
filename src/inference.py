import torch
import hydra
import glob
import imageio
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from diffusion_model.model import DiffusionModel


def save_image(filename, img):
    img = (img.clamp(-1, 1) + 1) / 2
    img = (img * 255).type(torch.uint8)
    imageio.imsave(filename, img)


def stack_samples(gen_samples, stack_dim):
    gen_samples = list(torch.split(gen_samples, 1, dim=1))
    for i in range(len(gen_samples)):
        gen_samples[i] = gen_samples[i].squeeze(1)
    return torch.cat(gen_samples, dim=stack_dim)


def save_gif(filename, gen_samples):
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
    gen_samples = (gen_samples * 255).type(torch.uint8)

    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)

    imageio.mimsave(filename, list(gen_samples), fps=5)


@hydra.main(version_base=None,
            config_path="../configs",
            config_name="inference")
def my_app(cfg: DictConfig):
    last_checkpoint = glob.glob(cfg.checkpoint_dir +
                                "*.ckpt")[cfg.index_checkpoint]

    model: LightningModule = DiffusionModel.load_from_checkpoint(
        last_checkpoint,
        t_range=cfg.model.t_range,
        img_dims=cfg.img_dims,
        backbone=cfg.backbone,
        attention=cfg.attention)
    model.eval()

    gif_shape = cfg.gif_shape
    sample_batch_size = gif_shape[0] * gif_shape[1]
    n_hold_final = 10

    # Generate samples from denoising process
    gen_samples = []
    x = torch.randn(
        (sample_batch_size, cfg.img_dims[0], cfg.img_dims[1], cfg.img_dims[2]))
    sample_steps = torch.arange(model.t_range - 1, 0, -1)

    for t in sample_steps:
        x = model.denoise_sample(x, t)
        if t % 50 == 0 or t == 1:
            print(t)
            gen_samples.append(x)

    for _ in range(n_hold_final):
        gen_samples.append(x)
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1],
                                      cfg.img_dims[1], cfg.img_dims[2],
                                      cfg.img_dims[0])

    x = x.moveaxis(1, 3)
    img = []
    for i in range(sample_batch_size):
        img.append(x[i])
    img = torch.cat(img, dim=1)

    save_image(cfg.checkpoint_dir + f"{cfg.name_img}.jpg", img)
    save_gif(cfg.checkpoint_dir + f"{cfg.name_img}.gif", gen_samples)


if __name__ == "__main__":
    my_app()