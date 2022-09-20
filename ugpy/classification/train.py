"""
Structure from https://github.com/Lightning-AI/lightning/issues/9252
"""

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from ugpy.classification.datasets import CropDataModule
from ugpy.classification.classifier import ImClassifier


def train_and_test_with_logger(wandb_logger):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    if wandb.config["deterministic"]:
        pl.seed_everything(wandb.config["seed"], workers=True)
    # initialise a datamodule
    data_module = CropDataModule(wandb.config)
    # set up the model
    model = ImClassifier(wandb.config)
    # callback to remember the model with the best validation loss
    checkpoint_callback = ModelCheckpoint(monitor='val_avg_loss', mode="min",
                                          dirpath=wandb.run.dir,
                                          filename="checkpoint_best_{val_avg_loss:.4f}_{epoch}")
    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=wandb.config["max_epochs"],
                         accelerator=wandb.config["accelerator"],
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback],
                         deterministic=wandb.config["deterministic"]
                         )
    # Train the model
    trainer.fit(model, data_module)
    # Evaluate the model on the held-out test set
    trainer.test(datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)
    # log data info and close wandb run
    wandb.config.update(data_module.info)
    wandb.finish()


if __name__ == "__main__":
    config_file = "../configs/train_config.yaml"
    # more info on wandb logger and pytorch lightning integration:
    # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.wandb.html
    # and some general info and sweeps:
    # https://dvelopery0115.github.io/2021/08/01/Introduction_to_W&B.html
    wandb_logger = WandbLogger(project="UGPy-SynapseClassifier",
                               job_type="train",
                               log_model=True,
                               config=config_file)
    train_and_test_with_logger(wandb_logger)

