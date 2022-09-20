import numpy as np
import unittest
from ugpy.utils.features import DistnceBS


# from ugpy.loader import load_centroids, load_labels
class TestDistnceBS(unittest.TestCase):
    def test_distance_to_nn1(self):
        """
        Tests without resolution
        """
        centroids = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 2]])
        dbs = DistnceBS(centroids)
        print(dbs.value)
        dx = np.array([1.0, 1.0, 2.0])
        print(dx)
        self.assertTrue(np.all(np.equal(dx, dbs.value)))

    def test_distance_to_nn2(self):
        """
        Tests with resolution
        """
        centroids = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 2]])
        resolution = [10, 1, 1]
        dbs = DistnceBS(centroids, resolution=resolution)
        print(dbs.value)
        dx = np.array([2.0, 10.0, 2.0])
        print(dx)
        self.assertTrue(np.all(np.equal(dx, dbs.value)))

    def test_histogram(self):
        """
        Tests histogram plotting... you have to look at it
        """
        centroids = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 2]])
        resolution = [10, 1, 1]
        dbs = DistnceBS(centroids, resolution=resolution)
        dbs.histogram(n_bins=3)
