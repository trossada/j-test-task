import numpy as np

from utils import incremental_mean_and_var


class FeatureScaler:
    """Standardize features by determining the distribution mean and standard deviation for each feature,
    subtracting the mean from each feature and dividing the values (mean is already subtracted) 
    of each feature by its standard deviation.

    The standard score of a sample `x` is calculated as:
        z = (x - u) / s
    where `u` is the mean of the training samples,
    and `s` is the standard deviation of the training samples.

    StandardSelector(scikit-learn) is used as a reference solution.
    """

    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = 0

    def fit(self, X):
        """Compute the mean and std to be used for later scaling.
        """

        self.mean_, self.var_, self.n_samples_seen_ = incremental_mean_and_var(X, self.mean_, self.var_, self.n_samples_seen_)
        self.scale_ = np.sqrt(self.var_)

    def transform(self, X):
        """Perform standardization by centering and scaling
        """

        X -= self.mean_
        X /= self.scale_
        return X



