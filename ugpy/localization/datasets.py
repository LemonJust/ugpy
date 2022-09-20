import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import pytorch_lightning as pl

from ugpy.classification.datasets import VolumeDataset


class VolumeDetectorDataset(VolumeDataset):
    """
    Crops volumes through the center of the synapse, remembers the center position
    Detector specifics adopted from https://www.kaggle.com/code/artgor/object-detection-with-pytorch-lightning/notebook
    """

    # what data type to use when converting images to tensors
    # TODO : compare float32 to float16
    image_dtype = torch.float32
    """
    nn.CrossEntropyLoss expects its label input to be of type torch.Long and not torch.Float.
    Note that this behavior is opposite to nn.BCELoss where target is expected to be of the same type as the input.
    """
    # TODO : so what type should I use ?!?!
    # what data type to use when converting labels to tensors
    label_dtype = torch.float32

    def __init__(self,
                 img, side, centroids, labels=None, numba=True):
        """
        Prepare data for localization.
        """
        super(VolumeDetectorDataset, self).__init__(img, side, centroids, labels=labels, numba=numba)
        self.bbox_centroids = None

    def _locate_syn_centers(self, drop_boarder=0):
        """
        Will tell when there are synapses inside the bboxes
        drop_boarder : centroids detected this many pixels close to the border to the image are not recorded
        """
        print("Running _locate_syn_centers")
        # number: number of centroids in the bbox
        # local: centroids in the bbox (relative to bbox center)
        # global: centroids in the bbox (in image coordinates)
        bbox_centroids = {'number': [], 'local': [], 'global': []}

        for centroid in self.centroids:
            min_z, min_y, min_x = centroid - self.side // 2 + drop_boarder
            max_z, max_y, max_x = centroid + self.side // 2 - drop_boarder
            in_z = (self.synapse_centroids[:, 0] >= min_z) & (self.synapse_centroids[:, 0] <= max_z)
            in_y = (self.synapse_centroids[:, 1] >= min_y) & (self.synapse_centroids[:, 1] <= max_y)
            in_x = (self.synapse_centroids[:, 2] >= min_x) & (self.synapse_centroids[:, 2] <= max_x)
            # for numpy array you can use & instead of np.logical_and
            in_bbox = in_z & in_y & in_x

            indices = np.where(in_bbox)[0]
            local_centroids = self.synapse_centroids[indices] - centroid
            global_centroids = self.synapse_centroids[indices]

            bbox_centroids['number'].append(len(indices))
            bbox_centroids['local'].append(local_centroids)
            bbox_centroids['global'].append(global_centroids)

            self.bbox_centroids = bbox_centroids
