"""
Builds a probability map for a given image.
Uses prob_map_config.yaml to initialise.
"""
import glob
from numba import njit
import numpy as np
import os
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

from ugpy.classification.classifier import ImClassifier
from ugpy.classification.datasets import CropProbMapModule
from ugpy.utils.data import split_to_rois, Image
from ugpy.utils.loader import load_image


def predict(config, model, data_module):
    if config["input_type"] == "two slices":
        img1 = torch.permute(data_module.pred_dataset[:][0], (1, 0, 2, 3)).to('cuda')
        img2 = torch.permute(data_module.pred_dataset[:][1], (1, 0, 2, 3)).to('cuda')
        prediction = torch.sigmoid(model((img1, img2)))
        prediction = prediction.to("cpu").numpy()
    elif config["input_type"] == "volume":
        img = torch.permute(data_module.pred_dataset[:], (1, 0, 2, 3, 4)).to('cuda')
        prediction = torch.sigmoid(model(img))
        prediction = prediction.to("cpu").numpy()
    else:
        raise ValueError(f"input_type can be 'two slices' or 'volume' only, but got "
                         f"{config['input_type']}")
    return prediction

@njit
def place_predictions_into_map(prob_map, centroids, prediction, scale):
    """
    Puts predictions into the corresponding spots in the image.
    :param prob_map: numpy array, the size of the image being processes
    :param centroids: numpy array, Nx3, coordinates of the pixels in the image in zyx order
    :param prediction: numpy array, predictions for each pixel, dtype float32
    :param scale: the number to multiply each prediction before turning it into int
    :return: updated prob_map
    """
    for i_pixel in range(len(prediction)):
        pixel = centroids[i_pixel]
        prob_map[pixel[0], pixel[1], pixel[2]] = int(prediction[i_pixel] * scale)
    return prob_map


def prob_map_with_logger():
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
                data_module = CropProbMapModule([roi], img, config=wandb.config)
                data_module.setup(stage='predict')
                prediction = predict(wandb.config, model, data_module)
                prob_map = place_predictions_into_map(prob_map,
                                                      data_module.pred_dataset.centroids,
                                                      prediction,
                                                      wandb.config["scale_prob"])
    # save image
    prob_image = Image(wandb.config["resolution"], img=prob_map.astype(np.int16))
    save_file = f'map_id_{wandb.run.id}_ckpt_{wandb.config["checkpoint_run_id"]}_{wandb.config["image_file"]}'
    save_file = os.path.join(wandb.config["save_dir"], save_file)
    print(f"Saving map as {save_file}")
    prob_image.imwrite(save_file)

    # close wandb run
    wandb.finish()


if __name__ == "__main__":
    config_file = "../configs/prob_map_config_crop.yaml"

    wandb_logger = WandbLogger(project="UGPy-SynapseClassifier-ProbMap",
                               job_type="prob_map",
                               config=config_file)
    prob_map_with_logger()
