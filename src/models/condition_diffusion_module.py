from typing import Any

import torch
import pyrootutils
from torch.optim import Optimizer, lr_scheduler

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models import DiffusionModule
from src.models.components import DiffusionModel


class ConditionDiffusionModule(DiffusionModule):

    def __init__(
        self,
        net: DiffusionModel,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
    ) -> None:
        super().__init__(net, optimizer, scheduler)

    def model_step(self, batch: Any):
        batch, label = batch
        preds, targets = self.forward(batch, cond=label)
        loss = self.criterion(preds, targets)
        return loss, preds, targets


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    root = pyrootutils.find_root(search_from=__file__,
                                 indicator=".project-root")
    print("root: ", root)
    config_path = str(root / "configs" / "model")

    @hydra.main(version_base=None,
                config_path=config_path,
                config_name="condition_diffusion_module.yaml")
    def main(cfg: DictConfig):
        cfg.net.denoise_model.n_classes = 2
        condition_diffusion_module: ConditionDiffusionModule = hydra.utils.instantiate(cfg)
        input = torch.randn(2, 1, 32, 32)
        cond = torch.Tensor([0, 1]).type(torch.int64)
        preds, targets = condition_diffusion_module(input, cond)
        print(preds.shape, targets.shape)

    main()