"""
All sort of visualization
"""
import napari.layers as nl
import napari
from napari_utils import FixedPoints

import skimage.data
import skimage.filters
from napari.types import PointsData
import pandas as pd
import numpy as np
import os
import tifffile as tif
import json
import csv

from magicgui import magicgui

import datetime
# from enum import Enum
from pathlib import Path


class ImagePointsView:
    def __init__(self, images=None, point_clouds=None, boxes=None, resolution=None):
        """
        img_dict is a dictionary with core.Image class.
        ptc_dict is a dictionary with is a core.Points class...
        """
        self.imgs = images
        self.ptcs = point_clouds
        self.boxs = boxes
        self.resolution = resolution

    def view_in_napari(self, img_cm=None, ptc_cm=None, box_cm=None):
        """
        Display image with the corresponding point cloud.
        img_cm, ptc_cm : colormap names (str) and colors (str) for display
        """
        # TODO : generate colors if not provided ? maybe with napari.utils.colormaps.label_colormap()
        # TODO : split indo add_ptc and add_ptcs etc

        with napari.gui_qt():
            viewer = napari.Viewer()

            if self.ptcs is not None:
                self.add_ptcs(viewer, ptc_cm)

            if self.imgs is not None:
                self.add_imgs(viewer, img_cm)

            if self.boxs is not None:
                self.add_boxs(viewer, box_cm)

    def add_ptcs(self, viewer, ptc_cm):
        i_ptc = 0
        for name, ptc in self.ptcs.items():
            viewer.add_layer(FixedPoints(
                ptc.zyx['pix'],
                ndim=3,
                size=2,
                edge_width=1,
                scale=self.resolution,
                name=name,
                face_color=ptc_cm[i_ptc]))
            i_ptc = i_ptc + 1

    def add_imgs(self, viewer, img_cm):
        i_img = 0
        for name, image in self.imgs.items():
            viewer.add_image(image.img,
                             scale=self.resolution,
                             name=name,
                             colormap=img_cm[i_img],
                             blending='additive')
            i_img = i_img + 1

    def add_boxs(self, viewer, box_cm):
        i_box = 0
        for name, boxes in self.boxs.items():
            viewer.add_layer(nl.Shapes(
                data=boxes.get_vertices(),
                ndim=3,
                shape_type='rectangle',
                scale=self.resolution,
                name=name,
                edge_width=0.5,
                edge_color=box_cm[i_box],
                face_color=[0] * 4))
            i_box = i_box + 1
