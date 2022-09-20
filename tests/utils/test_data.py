import numpy as np
import unittest
from ugpy.utils.data import prepare_shift


class TestFunctions(unittest.TestCase):
    def test_prepare_shift(self):
        shift_by = [2]
        shift_labels = [1]
        shift = prepare_shift(shift_by, shift_labels)
        self.assertListEqual(shift['shifts'], [[2, 0, 0], [-2, 0, 0],
                                               [0, 2, 0], [0, -2, 0],
                                               [0, 0, 2], [0, 0, -2]])
        self.assertListEqual(shift['labels'], [1, 1, 1, 1, 1, 1])

        shift_by = [2, 3]
        shift_labels = [1, 0]
        shift = prepare_shift(shift_by, shift_labels)
        self.assertListEqual(shift['shifts'], [[2, 0, 0], [-2, 0, 0],
                                               [0, 2, 0], [0, -2, 0],
                                               [0, 0, 2], [0, 0, -2],
                                               [3, 0, 0], [-3, 0, 0],
                                               [0, 3, 0], [0, -3, 0],
                                               [0, 0, 3], [0, 0, -3],
                                               ])
        self.assertListEqual(shift['labels'], [1, 1, 1, 1, 1, 1,
                                               0, 0, 0, 0, 0, 0])
