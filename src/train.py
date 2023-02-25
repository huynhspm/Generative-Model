import glob
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from models.diffusion import DiffusionModel
from utils.utils import instantiate_callbacks, instantiate_loggers
from typing import List


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def my_app(cfg: DictConfig):

    # Code for optionally loading model
    last_checkpoint = None

    # Create datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")

    # Create model
    if cfg.load_model:
        last_checkpoint = glob.glob(cfg.checkpoint_dir + "*.ckpt")[-1]
        model: LightningModule = instantiate(cfg.model,
                                             img_dims=datamodule.dims)
        model = model.load_from_checkpoint(last_checkpoint)
        # model = DiffusionModel.load_from_checkpoint(
        #     last_checkpoint,
        #     t_range=cfg.model.t_range,
        #     img_dims=datamodule.dims,
        #     channels=cfg.channels,
        #     backbone=cfg.model.backbone,
        #     n_layer_blocks=cfg.n_layer_blocks,
        #     channel_multipliers=cfg.channel_multipliers,
        #     attention=cfg.model.attention,
        #     attention_levels=cfg.attention_levels,
        #     n_attention_heads=cfg.n_attention_heads,
        #     n_attention_layers=cfg.n_attention_layers)
    else:
        model: LightningModule = instantiate(cfg.model,
                                             img_dims=datamodule.dims)
    print(f"Instantiating model <{cfg.model._target_}>")

    # Create callbacks
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Create logger
    logger: List[LightningLoggerBase] = instantiate_loggers(cfg.get("logger"))

    # Create trainer
    trainer: Trainer = instantiate(cfg.trainer,
                                   logger=logger,
                                   callbacks=callbacks,
                                   resume_from_checkpoint=last_checkpoint)
    print(f"Instantiating trainer <{cfg.trainer._target_}>")

    # Train model
    trainer.fit(model=model,
                datamodule=datamodule)


if __name__ == "__main__":
    my_app()
