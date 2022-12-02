import glob
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, loggers
from utils.utils import instantiate_callbacks
from typing import List


@hydra.main(version_base=None,
            config_path="../configs",
            config_name="train")
def my_app(cfg: DictConfig):

    # Code for optionally loading model
    pass_version = None
    last_checkpoint = None

    # Create datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")

    # Create model
    model: LightningModule = instantiate(cfg.model,
                                         img_dims=datamodule.dims)
    print(f"Instantiating model <{cfg.model._target_}>")

    if cfg.load_model:
        pass_version = cfg.load_version_num
        checkpoint_dir = f"{cfg.logger.save_dir}/{cfg.dataset_choice}/version_{cfg.load_version_num}/checkpoints/*.ckpt"
        last_checkpoint = glob.glob(checkpoint_dir)[-1]
        model = model.load_from_checkpoint(last_checkpoint)

    # Create callbacks
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Create logger
    logger: loggers = instantiate(cfg.logger,
                                  name=cfg.dataset_choice,
                                  version=pass_version)
    print(f"Instantiating logger <{cfg.logger._target_}>")

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
