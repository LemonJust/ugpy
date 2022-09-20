import numpy as np
import unittest
from ugpy.classification.datasets import VolumeDataset
from ugpy.utils.loader import load_data, load_image


class TestVolumeDataset(unittest.TestCase):
    # initialise a dataset
    data_dir = r"D:/code/repos/ugpy/data/test/gad1b"
    roi_id = '1-1VXC'
    centroids, labels, img = load_data(data_dir, roi_id)

    def test_add_shifted_positive_examples(self):
        # testing small volumes --> 7 pixel side
        side = 7

        # create VolumeDataset ______________________________________________
        vd = VolumeDataset(self.img, side, self.centroids, labels=self.labels,
                           numba=True)
        last_synapse_coordinates = vd.synapse_centroids[-1]
        # add shift with label "1"
        # shape before the shift
        nc0, _ = vd.centroids.shape
        ns0, _ = vd.synapse_centroids.shape
        # shift
        translation_list = [[1, 0, 0]]
        shifted_labels = [0]
        vd.add_shifted_positive_examples(translation_list, shifted_labels)
        # shape after the shift should change a certain way:
        nc1, _ = vd.centroids.shape
        ns1, _ = vd.synapse_centroids.shape
        self.assertEqual(nc0 + ns0, nc1)
        self.assertEqual(ns0, ns1)

        # centroids after translation :
        diff = vd.centroids[-1] - last_synapse_coordinates
        self.assertListEqual(translation_list[0], diff.tolist())

        # recreate VolumeDataset ______________________________________________
        vd = VolumeDataset(self.img, side, self.centroids, labels=self.labels,
                           numba=True)
        # add shift with label "0"
        # shape before the shift
        nc0, _ = vd.centroids.shape
        ns0, _ = vd.synapse_centroids.shape
        # shift
        translation_list = [[1, 0, 0]]
        shifted_labels = [1]
        vd.add_shifted_positive_examples(translation_list, shifted_labels)
        # shape after the shift should change a certain way:
        nc1, _ = vd.centroids.shape
        ns1, _ = vd.synapse_centroids.shape
        self.assertEqual(nc0 + ns0, nc1)
        self.assertEqual(2 * ns0, ns1)

        # recreate VolumeDataset ______________________________________________
        vd = VolumeDataset(self.img, side, self.centroids, labels=self.labels,
                           numba=True)
        last_synapse_coordinates = vd.synapse_centroids[-1]
        # add shift with labels [0,1]
        # shape before the shift
        nc0, _ = vd.centroids.shape
        ns0, _ = vd.synapse_centroids.shape
        # shift
        translation_list = [[1, 0, 0], [-1, 0, 0]]
        shifted_labels = [0, 1]
        vd.add_shifted_positive_examples(translation_list, shifted_labels)
        # shape after the shift should change a certain way:
        nc1, _ = vd.centroids.shape
        ns1, _ = vd.synapse_centroids.shape
        self.assertEqual(nc0 + 2*ns0, nc1)
        self.assertEqual(2 * ns0, ns1)

        # centroids after translation :
        diff = vd.centroids[-1] - last_synapse_coordinates
        self.assertListEqual(translation_list[1], diff.tolist())
        diff = vd.synapse_centroids[-1] - last_synapse_coordinates
        self.assertListEqual(translation_list[1], diff.tolist())
