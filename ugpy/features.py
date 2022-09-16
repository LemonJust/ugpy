"""
Structure from https://github.com/Lightning-AI/lightning/issues/9252
"""
import glob
import os

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from ugpy.preprocess import Image


# let's estimate the size of the synapse

# let's estimate the distance between synapses
class DistnceBS:
    """
    Distance Between Synapses:
    distance to the nearest neighbour in the SAME timepoint.
    """

    def __init__(self, centroids, resolution=[1, 1, 1], units='um'):
        """
        :param centroids: array with first point cloud with shape (n, 3)
        :type centroids: numpy.array
        :param resolution: resolution iin zyx order
        :type resolution: list
        """
        self.centroids = centroids
        self.resolution = resolution
        self.units = units

        self.value = self.distance_to_nn()

        self.mean = np.mean(self.value)
        self.median = np.median(self.value)
        self.min = np.min(self.value)
        self.max = np.max(self.value)

    def distance_to_nn(self):
        """
        Gets the distance for the nearest neighbour for each synapse
        :return: 1D array of distances in physical units
        :rtype: 1D numpy array
        """
        # move to physical units
        centroids = self.centroids * self.resolution
        centroids_kdt = cKDTree(centroids)
        # the first nearest neighbour is itself
        nn_levels = [2]
        dx, pairs = centroids_kdt.query(centroids, nn_levels)
        return np.squeeze(dx)

    def histogram(self, n_bins=50, roi_id=""):
        """
        Visualises the distance as a histogram
        """
        textstr = '\n'.join((
            f"min {self.min:.2f}",
            f"median {self.median:.2f}",
            f"mean {self.mean:.2f}",
            f"max {self.max:.2f}"
        ))

        fig = plt.figure(figsize=(5, 5), dpi=160)
        n, bins, _ = plt.hist(self.value, bins=n_bins)
        plt.xlabel(f"Distance, {self.units}")
        plt.ylabel(f"Num. Synapses")
        plt.title(f"Distance to Nearest Neighbour {roi_id}")
        # place a text box in upper left in axes coords
        plt.text(bins[n_bins // 2], np.max(n) * 0.95, textstr,
                 fontsize=12, verticalalignment='top')
        plt.show()

        return fig


"""
Helper Functions _______________________________________________________________
"""


def plot_histogram_of_distance_to_nn(roi_id):
    data_dir = r"D:\Code\repos\UGPy\data\test\gad1b"
    _, npz_file, csv_file = get_filenames(data_dir, roi_id)

    centroids = load_centroids(npz_file)
    labels = load_labels(npz_file, csv_file)

    synapses = centroids[labels.astype(bool)]
    resolution = [0.68, 0.23, 0.23]

    dbs = DistnceBS(synapses, units='pixels')
    dbs.histogram(roi_id=roi_id)


if __name__ == "__main__":
    from ugpy.loader import load_centroids, load_labels, get_filenames

    roi_id = "1-1WHA"
    plot_histogram_of_distance_to_nn(roi_id)
