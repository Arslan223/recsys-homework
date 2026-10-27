import numpy as np
from scipy import sparse
from .common import Recommender


class ItemKNN(Recommender):
    def __init__(self, neighbors=100, shrinkage=20):
        self.neighbors, self.shrinkage = neighbors, shrinkage

    def fit(self, positive):
        self.history = positive.astype(np.float64).tocsr()
        gram = (self.history.T @ self.history).toarray()
        norms = np.sqrt(np.diag(gram))
        sim = gram / np.maximum(norms[:, None] * norms[None, :], 1e-12)
        sim *= gram / (gram + self.shrinkage + 1e-12)
        np.fill_diagonal(sim, 0)

        idx = np.argsort(-sim, axis=0, kind='stable')[:self.neighbors]
        cols = np.broadcast_to(np.arange(sim.shape[1]), idx.shape)
        self.similarity = sparse.csr_matrix((sim[idx, cols].ravel(),
                                            (idx.ravel(), cols.ravel())), shape=sim.shape)
        self.similarity.eliminate_zeros()
        return self

    def score(self, users):
        return (self.history[users] @ self.similarity).toarray()
