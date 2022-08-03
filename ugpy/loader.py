"""
Classes to load the centroids and labels, also drops unsegmented
"""
import json
import numpy as np
import pandas as pd
import random
import tifffile as tif
from tqdm import tqdm
import warnings
import shutil
import scipy.ndimage as nd
import os
from pathlib import Path
from synspy.analyze.util import load_segment_status_from_csv


def _centroids_from_npz(npz_filename):
    """
    Loads centroids from npz and shifts it by slice origin to place it into SPIM voxel coords
    """
    parts = np.load(npz_filename)
    centroids = parts['centroids'].astype(np.int32)

    props = json.loads(parts['properties'].tostring().decode('utf8'))
    slice_origin = np.array(props['slice_origin'], dtype=np.int32)

    # convert cropped centroids back to full SPIM voxel coords
    return centroids + slice_origin


def _centroids_from_csv():
    """
    Loads centroids from csv
    """
    pass


def _labels_from_npz_and_csv(npz_filename, csv_filename):
    """
    Loads labels from npz and csv
    """
    parts = np.load(npz_filename)
    centroids = parts['centroids'].astype(np.int32)
    props = json.loads(parts['properties'].tostring().decode('utf8'))
    slice_origin = np.array(props['slice_origin'], dtype=np.int32)
    statuses, _ = load_segment_status_from_csv(centroids, slice_origin, csv_filename)

    # interpret status flag values
    is_synapse = (statuses == 7).astype(bool)

    return is_synapse


def load_centroids(npz_filename):
    """
    Loads centroids from npz or csv
    """
    return _centroids_from_npz(npz_filename)


def load_labels(npz_filename, csv_filename):
    """
    Loads labels from npz and csv
    """
    return _labels_from_npz_and_csv(npz_filename, csv_filename)


def drop_unsegmented(centroids, labels, x_max=500, z_max=90):
    """
    Gets rid of centroids and labels in the unsegmented area.
    Keeping only stuff above Z = 90 and on the left side of X = 500.
    Currently that's what we have unsegmented for Gad1b::GFP sometimes.

    Parameters:
    x_max,z_max (int) : max x and z values (in pixels) to keep
    """
    z_coord = centroids[:, 0]
    x_coord = centroids[:, 2]

    in_z = z_coord <= z_max
    in_x = x_coord <= x_max
    in_roi = np.logical_and(in_z, in_x)

    centroids = centroids[in_roi, :]
    labels = labels[in_roi]

    return centroids, labels


def load_image(img_file):
    """
    Just a wrapper for tif.imread
    """
    img = tif.imread(img_file)
    return img


def get_filenames(data_dir, roi_id):

    data_folder = Path(f"{data_dir}/{roi_id}")
    img_files = [file for file in data_folder.glob("*.tiff")]
    npz_files = [file for file in data_folder.glob("*.npz")]
    csv_files = [file for file in data_folder.glob("*.csv")]

    assert len(img_files) == 1, f"Should be exactly 1 tiff file in {data_folder}, but found {len(img_files)}"
    assert len(npz_files) == 1, f"Should be exactly 1 npz file in {data_folder}, but found {len(npz_files)}"
    assert len(csv_files) == 1, f"Should be exactly 1 csv file in {data_folder}, but found {len(csv_files)}"

    return img_files[0], npz_files[0], csv_files[0]

def load_data(data_dir, roi_id):
    """
    Loads the centroids, labels and images from which we will crop
    """

    img_filename, npz_filename, csv_filename = get_filenames(data_dir, roi_id)

    centroids = load_centroids(npz_filename)
    labels = load_labels(npz_filename, csv_filename)

    warnings.warn("Assuming area with x>500 and z>90 is unsegmented. "
                  "Dropping all the centroids with x_max=500 z_max=90. ")
    centroids, labels = drop_unsegmented(centroids, labels, x_max=500, z_max=90)
    img = load_image(img_filename)

    return centroids, labels, img

# TODO: Create loader for sycatch type files and add tag to load_labels, load_centroids
#
# """
# Takes care of loading centroids and labels from how they are stored by Sycatch.
# """
#
# # def __init__(self):
#
# def _load_centroids_sc(self):
#     """
#     Loads centroids from csv
#     """
#     pass
#
# def _load_labels_sc(self):
#     """
#     Loads labels from npz and csv
#     """
#     pass
