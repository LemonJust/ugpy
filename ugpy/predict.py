"""
THIS FILE IS UNFINISHED !!!
just a template ...
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
from data import split_to_rois, get_image_shape, Image
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
    model = ImClassifier.load_from_checkpoint(checkpoint_path, config=wandb.config)
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

    for i_roi, roi in enumerate(rois):
        print(f"Calculating probability for roi {i_roi}/{len(rois)}")
        data_module = TwoSlicesProbMapModule([roi], img=img, num_workers=wandb.config["num_workers"])
        prediction = trainer.predict(model, datamodule=data_module)[0]
        for i, pixel in enumerate(data_module.pred_dataset.centroids):
            prob_map[pixel[0], pixel[1], pixel[2]] = int(prediction[i] * wandb.config["scale_prob"])
    print(f"Writing image")
    prob_image = Image(wandb.config["resolution"], img=prob_map.astype(np.int16))
    save_file = f'probmap{wandb.config["save_tag"]}_{wandb.config["checkpoint_run_id"]}_{wandb.config["image_file"]}'
    prob_image.imwrite(os.path.join(wandb.config["save_dir"], save_file))

    # close wandb run

    wandb.finish()


def predict(project_name):
    wandb_logger = WandbLogger(project=project_name, job_type='predict')
    accelerator = 'gpu'

    img_file = "D:/Code/repos/UGPy/data/predict/prob_map/1-20FJ.tif"
    # use chunks to draw probability map in pieces
    zyx_chunks = [3, 3, 3]
    zyx_padding = [10, 10, 10]
    rois = split_to_rois(img_file, zyx_chunks, zyx_padding)
    num_workers = 0
    batch_size = 50000
    data_module = TwoSlicesProbMapModule(img_file, [rois[0]], batch_size, num_workers=num_workers)

    # # get the check point
    # # TODO : try wandb restore
    run_id = "3m51q4ne"
    checkpoint_path = glob.glob("D:/Code/repos/UGPy/ugpy/wandb/*3m51q4ne/files/*.ckpt")[0]
    # glob.glob("D:/Code/repos/UGPy/ugpy/wandb/*3m51q4ne/files/*.ckpt")
    print(f"Using checkpoint : {checkpoint_path}")
    # set up the model
    # TODO : make it so that I don't need to set up the loss ?
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5), reduction='mean')
    model = SynapseClassifier.load_from_checkpoint(TwoBranchConv2d(), criterion, checkpoint_path)

    # Initialize a trainer
    if accelerator == 'gpu':
        trainer = pl.Trainer(max_epochs=1,
                             accelerator='gpu', devices=1,
                             logger=wandb_logger)

    elif accelerator == 'cpu':
        print("Using cpu only")
        trainer = pl.Trainer(max_epochs=1,
                             accelerator='cpu',
                             logger=wandb_logger)

    wandb.config["accelerator"] = accelerator

    prediction = trainer.predict(model, datamodule=data_module)

    wandb.config["batch_size"] = batch_size
    wandb.config["num_pred"] = len(data_module.pred_dataset)
    wandb.config["num_workers"] = num_workers
    wandb.config["accelerator"] = accelerator

    # close wandb run
    wandb.finish()

    # seed = 222
    # img_file = "D:/Code/repos/UGPy/data/predict/prob_map/raw/1-20FJ.tif"
    # run_id = "3m51q4ne"
    # project_name = 'UGPy-SynapseClassifier'


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
from data import split_to_rois, get_image_shape, Image
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
    model = ImClassifier.load_from_checkpoint(checkpoint_path, config=wandb.config)
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
            data_module = TwoSlicesProbMapModule([roi], img=img, num_workers=wandb.config["num_workers"])
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
    config_file = "predict_config2.yaml"

    wandb_logger = WandbLogger(project="UGPy-SynapseClassifier",
                               job_type="prob_map",
                               config=config_file)
    prob_map_with_logger(wandb_logger)

