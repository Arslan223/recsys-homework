import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .common import Recommender


class EASE(Recommender):
    def __init__(self, regularization=100.):
        self.regularization = regularization

    def fit(self, positive):
        self.history = positive.astype(np.float64).tocsr()
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::gram.shape[0] + 1] += self.regularization
        precision = cho_solve(cho_factor(gram, lower=True, check_finite=False),
                              np.eye(gram.shape[0]), check_finite=False)
        self.coefficients = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.coefficients, 0.)
        return self

    def score(self, users):
        return self.history[users] @ self.coefficients
