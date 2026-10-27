import numpy as np
from scipy import sparse
from .ials import ImplicitALS


class CustomTargetALS(ImplicitALS):
    def __init__(self, half_life_days=365., **kwargs):
        super().__init__(**kwargs)
        self.half_life_days = half_life_days

    def fit(self, ratings, timestamps, cutoff):
        observed = ratings.copy().astype(np.float64)
        observed.data[:] = 1.
        if not (np.array_equal(ratings.indices, timestamps.indices)
                and np.array_equal(ratings.indptr, timestamps.indptr)):
            raise ValueError('Ratings and timestamps must have matching sparsity')
        age_days = (cutoff - timestamps.data) / 86400.
        if np.any(age_days < 0):
            raise ValueError('Future interaction in training data')
        decay = .5 + .5 * np.exp2(-age_days / self.half_life_days)
        target = ratings.copy().astype(np.float64)
        target.data = ((ratings.data - 3.) / 2.) * decay
        self.target_summary = {key: float(value) for key, value in zip(
            ['min', 'mean', 'max'], [target.data.min(), target.data.mean(), target.data.max()])}
        return self.fit_target(observed, target)
