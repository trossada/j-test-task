from unittest import TestCase

import numpy as np

from utils import incremental_mean_and_var, get_max_feature_index_and_abs_mean_diff


class TestUtils(TestCase):

    def test_incremental_mean_and_var(self):
        X, Y = np.arange(10).reshape(2, 5), np.arange(10, 20).reshape(2,5)
        X_mean, Y_mean, XY_mean = np.mean(X, axis=0), np.mean(Y, axis=0), np.mean(np.vstack((X, Y)), axis=0)

        new_X_mean, new_X_var, new_X_sample_count = incremental_mean_and_var(X)
        new_Y_mean, new_Y_var, new_Y_sample_count = incremental_mean_and_var(Y)
        new_XY_mean, new_XY_var, new_XY_sample_count = incremental_mean_and_var(Y, new_X_mean, new_X_var, new_X_sample_count)
        new_YX_mean, new_YX_var, new_YX_sample_count = incremental_mean_and_var(X, new_Y_mean, new_Y_var, new_Y_sample_count)

        self.assertTrue(np.array_equal(X_mean, new_X_mean))
        self.assertTrue(np.array_equal(Y_mean, new_Y_mean))
        self.assertTrue(np.array_equal(XY_mean, new_XY_mean))
        self.assertTrue(np.array_equal(XY_mean, new_YX_mean))

    def test_incremental_max_feature_index_and_abs_mean_diff(self):
        X = np.array(
            [[1, 5, 0, 0, 0],
             [2, 0, 5, 0, 8],
             [3, 1, 1, 3, 1]]
        )

        # mean:  2 2 2 1 3
        # i:         1 4 0
        # max:     _ 5 8 3
        # mean[i]:   2 3 2
        #            _____
        # diff:      3 5 1

        mean = incremental_mean_and_var(X)[0]

        X_mfi_expect = np.array([1, 4, 0])
        X_mfamd_expect = np.array([3, 5, 1])

        X_mfi, X_mfamd = get_max_feature_index_and_abs_mean_diff(X, mean)
        self.assertTrue(np.array_equal(X_mfi_expect, X_mfi))
        self.assertTrue(np.array_equal(X_mfamd_expect, X_mfamd))