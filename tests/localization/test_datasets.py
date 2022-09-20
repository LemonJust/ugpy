import numpy as np
import unittest
from ugpy.localization.datasets import VolumeDetectorDataset
from ugpy.utils.loader import load_data, load_image


class TestVolumeDetectorDataset(unittest.TestCase):
    # initialise a dataset
    data_dir = r"D:/code/repos/ugpy/data/test/gad1b"
    roi_id = '1-1VXC'
    centroids, labels, img = load_data(data_dir, roi_id)

    def test_locate_syn_centers_number(self):
        """
        After looking at the distance between synapses in our data ,
        we know that the min distance between two synapses is 3 pixels.
        Side 7 should allow for two synapses in the box,
        side 5 should not get more than 1 synapse per box.

        If these tests fail, the code might be okay,
        but it would indicate that our data has changed.

        Intended use: _locate_syn_centers with side = 7, drop_boarder = 1
        """
        # testing small volumes --> 7 pixel side
        side = 7
        vdd = VolumeDetectorDataset(self.img, side, self.centroids, labels=self.labels,
                                    numba=True)
        vdd._locate_syn_centers()
        self.assertEqual(np.max(vdd.bbox_centroids['number']), 2)
        self.assertEqual(np.min(vdd.bbox_centroids['number']), 0)

        vdd._locate_syn_centers(drop_boarder=1)
        self.assertEqual(np.max(vdd.bbox_centroids['number']), 1)
        self.assertEqual(np.min(vdd.bbox_centroids['number']), 0)

        # testing small volumes --> 5 pixel side
        side = 5
        vdd = VolumeDetectorDataset(self.img, side, self.centroids, labels=self.labels,
                                    numba=True)
        vdd._locate_syn_centers()
        self.assertEqual(np.max(vdd.bbox_centroids['number']), 1)
        self.assertEqual(np.min(vdd.bbox_centroids['number']), 0)

    def test_locate_syn_centers_centroids(self):
        """
        making sure the centroids are recorded correctly
        """
        # testing small volumes --> 7 pixel side
        side = 7
        vdd = VolumeDetectorDataset(self.img, side, self.centroids, labels=self.labels,
                                    numba=True)
        vdd._locate_syn_centers(drop_boarder=1)

        # all centroids in the box should be at the box center
        syn_in_box = np.where(np.array(vdd.bbox_centroids['number']) > 0)[0]
        syn_centroids = [vdd.bbox_centroids['local'][i] for i in syn_in_box]
        self.assertTrue(np.all([centroid == [[0, 0, 0]] for centroid in syn_centroids]))

    def test_locate_syn_centers_centroids(self):
        """
        making sure the centroids are recorded correctly
        """
        # testing small volumes --> 7 pixel side
        side = 7
        vdd = VolumeDetectorDataset(self.img, side, self.centroids, labels=self.labels,
                                    numba=True)
        vdd._locate_syn_centers(drop_boarder=1)

        # all centroids in the box should be at the box center
        syn_in_box = np.where(np.array(vdd.bbox_centroids['number']) > 0)[0]
        syn_centroids = [vdd.bbox_centroids['local'][i] for i in syn_in_box]
        self.assertTrue(np.all([centroid == [[0, 0, 0]] for centroid in syn_centroids]))
