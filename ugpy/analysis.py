"""
Analysis of the probability maps and other things
"""

import numpy as np
from data import Image
from dipy.align.reslice import affine_transform


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
