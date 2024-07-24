from typing import Any

import torch
import pyrootutils
from torch.optim import Optimizer, lr_scheduler

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.diffusion import DiffusionModule
from src.models.diffusion.net import ConditionDiffusionModel


class ConditionDiffusionModule(DiffusionModule):

    def __init__(self,
                 net: ConditionDiffusionModel,
                 optimizer: Optimizer,
                 scheduler: lr_scheduler,
                 use_ema: bool = False,
                 compile: bool = False) -> None: 
        
        super().__init__(net, optimizer, scheduler, use_ema, compile)

    def model_step(self, batch: Any):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
        """

        batch, cond = batch
        preds, targets = self.forward(batch, cond=cond)
        loss = self.criterion(preds, targets)
        return loss


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model" / "diffusion")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="condition_diffusion_module.yaml")
    def main(cfg: DictConfig):
        cfg['net']['n_train_steps'] = 1000
        cfg['net']['sampler']['n_train_steps'] = 1000
        cfg['net']['img_dims'] = [1, 32, 32]
        cfg['net']['denoise_net']['d_cond_image'] = 1
        cfg['net']['label_embedder'] = {
            '_target_': 'src.models.components.embeds.LabelEmbedder',
            'n_classes': 2,
            'd_embed': 256,
        }
        # print(cfg)

        condition_diffusion_module: ConditionDiffusionModule = hydra.utils.instantiate(
            cfg)

        x = torch.randn(2, 1, 32, 32)
        cond = {
            'label':
            torch.randint(0, cfg['net']['label_embedder']['n_classes'], (2, )),
            'image':
            torch.rand_like(x),
        }
        pred, target = condition_diffusion_module(x, cond)
        print('***** CONDITION DIFFUSION MODEL MODULE *****')
        print('Input:', x.shape)
        print('Prediction:', pred.shape)
        print('Target:', target.shape)

    main()