import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from .common import Recommender


class ImplicitALS(Recommender):
    def __init__(self, factors=32, alpha=20., regularization=10., iterations=12, seed=42):
        self.factors, self.alpha = factors, alpha
        self.regularization, self.iterations, self.seed = regularization, iterations, seed

    @staticmethod
    def _step(observed, target, fixed, regularization, alpha):
        fixed_gram = fixed.T @ fixed + regularization * np.eye(fixed.shape[1])
        updated = np.zeros((observed.shape[0], fixed.shape[1]))
        for row in range(observed.shape[0]):
            start, end = observed.indptr[row:row + 2]
            indices = observed.indices[start:end]
            selected = fixed[indices]

            values = target.data[start:end]
            normal = fixed_gram + alpha * (selected.T @ selected)
            rhs = (1. + alpha) * (selected.T @ values)
            updated[row] = cho_solve(cho_factor(normal, lower=True, check_finite=False),
                                     rhs, check_finite=False)
        return updated

    def fit_target(self, observed, target):
        observed = observed.astype(np.float64).tocsr()
        target = target.astype(np.float64).tocsr()
        observed.sort_indices(); target.sort_indices()
        if not (np.array_equal(observed.indptr, target.indptr)
                and np.array_equal(observed.indices, target.indices)):
            raise ValueError('Target must retain all observed entries, including zeros')
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, .05, (observed.shape[1], self.factors))
        obs_t, tar_t = observed.T.tocsr(), target.T.tocsr()
        for _ in range(self.iterations):
            self.users = self._step(observed, target, self.items, self.regularization, self.alpha)
            self.items = self._step(obs_t, tar_t, self.users, self.regularization, self.alpha)
        return self

    def fit(self, ratings):
        observed = ratings.copy().astype(np.float64)
        observed.data[:] = 1.
        target = ratings.copy().astype(np.float64)
        target.data = (target.data >= 4).astype(float)
        return self.fit_target(observed, target)

    def score(self, users):
        return self.users[users] @ self.items.T
