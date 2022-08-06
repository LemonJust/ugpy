"""
Structure from https://github.com/Lightning-AI/lightning/issues/9252
"""
import glob
import os

import numpy as np
import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb

from classifier import ImClassifier

from datasets import TwoSlicesDataModule, TwoSlicesProbMapModule
from models import TwoBranchConv2d
from preprocess import split_to_rois, get_image_shape, Image
from loader import load_image


def prob_map_with_logger(wandb_logger):
    """
    Creates a probability map , working in chunks. Chunks are currently set up to work on my machine,
    but might need to be entered by user in the future.
    """
    # TODO : try wandb restore instead of a local directory
    # TODO : use the proper path merging
    checkpoint_path = glob.glob(
        f'{wandb.config["checkpoint_dir"]}/*{wandb.config["checkpoint_run_id"]}/files/*.ckpt')[0]
    print(f"Using checkpoint file:\n {checkpoint_path}")

    # set up the model
    model = ImClassifier.load_from_checkpoint(checkpoint_path, config=wandb.config).cuda()
    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=1,
                         accelerator=wandb.config["accelerator"],
                         logger=wandb_logger
                         )

    # use chunks to draw probability map in pieces
    image_file = os.path.join(wandb.config["image_dir"], wandb.config["image_file"])
    rois = split_to_rois(image_file, wandb.config["split"], wandb.config["margin"])

    img = load_image(image_file)
    prob_map = np.zeros(img.shape, dtype=np.int16)
    # predict rois
    for i_roi, roi in enumerate(rois):
        print(f"Calculating probability for roi {i_roi}/{len(rois)}")
        if i_roi in wandb.config["skip_rois"]:
            print("Skipped")
        else:
            data_module = TwoSlicesProbMapModule([roi], img, config=wandb.config)
            prediction = trainer.predict(model, datamodule=data_module)[0]
            for i, pixel in enumerate(data_module.pred_dataset.centroids):
                prob_map[pixel[0], pixel[1], pixel[2]] = int(prediction[i] * wandb.config["scale_prob"])
    # save image
    prob_image = Image(wandb.config["resolution"], img=prob_map.astype(np.int16))
    save_file = f'probmap_id{wandb.run.id}_ckpt{wandb.config["checkpoint_run_id"]}_{wandb.config["image_file"]}'
    save_file = os.path.join(wandb.config["save_dir"], save_file)
    print(f"Saving map as {save_file}")
    prob_image.imwrite(save_file)

    # close wandb run
    wandb.finish()


if __name__ == "__main__":
    config_file = "prob_map_config.yaml"

    wandb_logger = WandbLogger(project="UGPy-SynapseClassifier",
                               job_type="prob_map",
                               config=config_file)
    prob_map_with_logger(wandb_logger)
