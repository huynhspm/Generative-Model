import matplotlib.pyplot as plt
import torch
import glob
import imageio
from models.model import DiffusionModel

DIFFUSION_STEPS = 1000
DATASET_CHOICES = "GENDER"
LOAD_VERSION_NUM = 3
IMG_DIMS = (3, 64, 64)
DIR = f"./src/tensorboard/{DATASET_CHOICES}/version_{LOAD_VERSION_NUM}/"

last_checkpoint = glob.glob(DIR + "checkpoints/*.ckpt")[-1]
model = DiffusionModel.load_from_checkpoint(
    last_checkpoint, t_range=DIFFUSION_STEPS, img_dims=IMG_DIMS)
model.eval()

gif_shape = [1, 9]
sample_batch_size = gif_shape[0] * gif_shape[1]
n_hold_final = 10

# Generate samples from denoising process
gen_samples = []
x = torch.randn((sample_batch_size, IMG_DIMS[0],
                IMG_DIMS[1], IMG_DIMS[2]))
sample_steps = torch.arange(model.t_range-1, 0, -1)
image = []
for t in sample_steps:
    x = model.denoise_sample(x, t)
    if t % 50 == 0 or t == 1:
        print(t)
        gen_samples.append(x)

for _ in range(n_hold_final):
    gen_samples.append(x)
gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2


# Process samples and save as gif
gen_samples = (gen_samples * 255).type(torch.uint8)
gen_samples = gen_samples.reshape(-1, gif_shape[0],
                                  gif_shape[1], IMG_DIMS[1], IMG_DIMS[2], IMG_DIMS[0])


def stack_samples(gen_samples, stack_dim):
    gen_samples = list(torch.split(gen_samples, 1, dim=1))
    for i in range(len(gen_samples)):
        gen_samples[i] = gen_samples[i].squeeze(1)
    return torch.cat(gen_samples, dim=stack_dim)


gen_samples = stack_samples(gen_samples, 2)
gen_samples = stack_samples(gen_samples, 2)

imageio.mimsave(DIR + "pred3.gif", list(gen_samples), fps=5)
