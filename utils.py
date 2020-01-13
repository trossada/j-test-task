import numpy as np


def incremental_mean_and_var(X, last_mean=None, last_variance=None, last_sample_count=None, ignore_var=False):
    """Calculate (or update) mean and var
    """

    if last_sample_count is None or (type(last_sample_count) == int and last_sample_count == 0):
        last_sample_count = np.repeat(0, X.shape[1]).astype(np.int64, copy=False)
        last_variance = 0
        last_sum = 0
    else:
        last_sum = last_mean * last_sample_count

    new_sum = np.sum(X, axis=0)
    new_sample_count = np.repeat(X.shape[0], X.shape[1]).astype(np.int64, copy=False)
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if ignore_var:
        updated_variance = None
    else:
        new_unnormalized_variance = np.nanvar(X, axis=0) * new_sample_count
        last_unnormalized_variance = last_variance * last_sample_count
        with np.errstate(divide='ignore', invalid='ignore'):
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = (
                    last_unnormalized_variance + new_unnormalized_variance +
                    last_over_new_count / updated_sample_count *
                    (last_sum / last_over_new_count - new_sum) ** 2)

        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count
    return updated_mean, updated_variance, updated_sample_count


def get_max_feature_index_and_abs_mean_diff(X, mean):
    """Get index `i` of max value and calculate absolute deviation of this max value and average value of feature[i]
    """

    max_feature_index = np.argmax(X, axis=1)
    max_feature = np.max(X, axis=1)
    max_feature_abs_mean_diff = np.abs(max_feature - mean[max_feature_index])
    return max_feature_index, max_feature_abs_mean_diff