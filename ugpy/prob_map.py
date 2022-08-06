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
from numba import njit


@njit
def place_predictions_into_map(prob_map, centroids, prediction, scale):
    for i_pixel in range(len(prediction)):
        pixel = centroids[i_pixel]
        prob_map[pixel[0], pixel[1], pixel[2]] = int(prediction[i_pixel] * scale)
    return prob_map


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
    model = ImClassifier.load_from_checkpoint(checkpoint_path, config=wandb.config).to('cuda')
    model.eval()
    # Initialize a trainer

    # use chunks to draw probability map in pieces
    image_file = os.path.join(wandb.config["image_dir"], wandb.config["image_file"])
    rois = split_to_rois(image_file, wandb.config["split"], wandb.config["margin"])

    img = load_image(image_file)
    prob_map = np.zeros(img.shape, dtype=np.int16)
    # predict rois
    with torch.no_grad():
        for i_roi, roi in enumerate(rois):
            print(f"Calculating probability for roi {i_roi}/{len(rois)}")
            if i_roi in wandb.config["skip_rois"]:
                print("Skipped")
            else:
                data_module = TwoSlicesProbMapModule([roi], img, config=wandb.config)
                print("creating dataset")
                data_module.setup(stage='predict')
                print("moving to GPU")
                img1 = torch.permute(data_module.pred_dataset[:][0], (1, 0, 2, 3)).to('cuda')
                img2 = torch.permute(data_module.pred_dataset[:][1], (1, 0, 2, 3)).to('cuda')
                # print("creating dataloader")
                # dataloader = data_module.predict_dataloader()
                # print("getting batch")
                # batch = next(iter(dataloader))
                print("predicting")
                prediction = torch.sigmoid(model((img1, img2)))
                print("moving predictions to cpu")
                prediction = prediction.to("cpu").numpy()
                print("putting into the map")

                prob_map = place_predictions_into_map(prob_map,
                                                      data_module.pred_dataset.centroids,
                                                      prediction,
                                                      wandb.config["scale_prob"])
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
