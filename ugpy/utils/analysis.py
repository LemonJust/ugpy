"""
Analysis of the probability maps and other things
"""

import numpy as np
from pathlib import Path
from data import Image
from dipy.align.reslice import affine_transform
import skimage as sm


def apply_transformation():
    fixed_file = ''
    moving_file = ''
    resolution = []
    output_shape = Image.get_image_shape(fixed_file)
    moving = Image(resolution, filename=moving_file)
    matrix = np.array(
        [[0.9997562490148609, 0.022054010835901916, 0.0010310973672847548, -2.8755508365952753],
         [-0.022027096995440473, 0.9995309436192478, -0.02127674189275605, 2.4039545698438025],
         [-0.0014998512207409832, 0.021248843584238262, 0.9997730928028863, -3.7580929915470414],
         [0, 0, 0, 1]]
    )

    affine_transform(input, matrix, output_shape=output_shape)


def label_regions(label_image):
    """
    Splits all regions into separate synapses. Wrapper for skimage.measure.label .
    :param label_image: Image to be labeled
    :type label_image: ndarray of dtype int
    :return:
    :rtype:
    """
    labeled_image, num_labels = sm.measure.label(label_image, background=None, return_num=True, connectivity=None)
    return labeled_image, num_labels


# ________________________________________________________________________________________________________________

def process_prob_map():
    """
    Adapted from scikit-image.org regionprops_table tutorial.

    Also see:
    scikit-image.org 'Label image regions' tutorial: https://tinyurl.com/2ky2wuwf
    And there are more features there , that I don't use, like cool plotting !
    """

    from skimage import data, util, measure
    import pandas as pd

    data_dir = r"D:\Code\repos\UGPy\data\changes\learners\1-23C4"

    prob_map = Image([0.68, 0.23, 0.23],
                     filename=Path(data_dir, r"prob_map\crops\map_id_mz987f8m_ckpt_1ibq4uut_1-23E4_left_main.tif"))
    # image = Image([0.68, 0.23, 0.23],
    #               filename=Path(data_dir, r"crops\1-23E4_left_main.tif"))

    threshold = 0
    label_image = measure.label(prob_map.img > threshold, connectivity=prob_map.img.ndim)
    props = measure.regionprops_table(label_image, prob_map.img,
                                      properties=['label', 'centroid', 'area', 'area_filled'])
    prob_df = pd.DataFrame(props)

    # props = measure.regionprops_table(label_image, image.img, properties=['label'])
    # raw_df = pd.DataFrame(props)

    return Image([0.68, 0.23, 0.23], img=label_image), prob_df


if __name__ == "__main__":
    data_dir = r"D:\Code\repos\UGPy\data\changes\learners\1-23C4"
    img, data_df = process_prob_map()
    img.imwrite(Path(data_dir, "prob_map_labeled_1-23E4_left_main.tif"))
    data_df.to_csv(Path(data_dir, "prob_map_features_1-23E4_left_main.csv"), index=False)
