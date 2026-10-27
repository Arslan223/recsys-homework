import unittest
import numpy as np
import torch
from scipy import sparse
from models import EASE, ItemKNN, ImplicitALS, CustomTargetALS, FactorizationMachine
from models.fm import _FM
from models.common import Recommender
from evaluation import ranking_metrics


class CorrectnessTests(unittest.TestCase):
    def test_als_step_matches_dense_weighted_least_squares(self):
        obs = sparse.csr_matrix([[1., 0, 1], [0, 1, 0]])
        target = obs.copy(); target.data[:] = [.7, -.4, 0.]
        fixed = np.array([[.2, .4], [-.1, .7], [.8, -.3]])
        actual = ImplicitALS._step(obs, target, fixed, .6, 3.)
        for u in range(2):
            c = np.diag(1 + 3 * obs.toarray()[u])
            expected = np.linalg.solve(fixed.T @ c @ fixed + .6 * np.eye(2),
                                        fixed.T @ c @ target.toarray()[u])
            np.testing.assert_allclose(actual[u], expected, atol=1e-12)

    def test_ease_matches_constrained_ridge_per_column(self):
        x = np.array([[1., 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        model = EASE(2.).fit(sparse.csr_matrix(x))
        for j in range(3):
            others = np.arange(3) != j
            expected = np.linalg.solve(x[:, others].T @ x[:, others] + 2 * np.eye(2), x[:, others].T @ x[:, j])
            np.testing.assert_allclose(model.coefficients[others, j], expected, atol=1e-12)
        np.testing.assert_array_equal(model.coefficients.diagonal(), 0)

    def test_fm_matches_explicit_pair_sum_and_aggregated_scoring(self):
        torch.manual_seed(1)
        net = _FM(10, 3)
        f = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]])
        with torch.no_grad():
            expected = net.bias + net.linear(f).sum((1, 2))
            for a in range(4):
                for b in range(a + 1, 4):
                    expected = expected + (net.factors(f[:, a]) * net.factors(f[:, b])).sum(1)
            np.testing.assert_allclose(net(f).numpy(), expected.numpy(), atol=1e-7)
            u, ub = net.aggregate(f[:, :2]); i, ib = net.aggregate(f[:, 2:])
            aggregate = (u * i).sum(1) + ub + ib + net.bias
            np.testing.assert_allclose(aggregate.numpy(), net(f).numpy(), atol=1e-7)

    def test_metrics_known_ranking(self):
        recommended = np.arange(10)
        ndcg, recall, hit = ranking_metrics(recommended, [0, 2, 11])
        expected = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3) + 1 / np.log2(4))
        self.assertAlmostEqual(ndcg, expected)
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertEqual(hit, 1.)
        self.assertEqual(ranking_metrics(recommended, [20])[0], 0.)

    def test_seen_filter_and_stable_ties(self):
        class Dummy(Recommender):
            def score(self, users):
                return np.ones((len(users), 12))
        seen = sparse.csr_matrix(([1., 1.], ([0, 0], [0, 3])), shape=(1, 12))
        np.testing.assert_array_equal(Dummy().recommend([0], seen)[0], [1, 2, 4, 5, 6, 7, 8, 9, 10, 11])

    def test_knn_cosine_and_self_exclusion(self):
        x = sparse.csr_matrix([[1., 1, 0], [1, 0, 1], [1, 1, 0]])
        model = ItemKNN(neighbors=2, shrinkage=0).fit(x)
        s = model.similarity.toarray()
        self.assertAlmostEqual(s[0, 1], 2 / np.sqrt(6))
        np.testing.assert_array_equal(s.diagonal(), 0)
        np.testing.assert_allclose(model.score([0]), x[[0]].toarray() @ s)

    def test_custom_target_changes_rhs_not_confidence(self):
        class Capture(CustomTargetALS):
            def fit_target(self, observed, target):
                self.observed, self.target = observed, target
                return self
        ratings = sparse.csr_matrix([[1., 3., 5.]])
        times = ratings.copy(); times.data[:] = [86400., 86400., 86400.]
        model = Capture(half_life_days=1).fit(ratings, times, 2 * 86400.)
        np.testing.assert_array_equal(model.observed.data, 1.)
        np.testing.assert_allclose(model.target.data, [-.75, 0, .75])
        with self.assertRaises(ValueError):
            Capture().fit(ratings, times, 0)

    def test_als_reproducible(self):
        r = sparse.csr_matrix([[5., 2, 0], [0, 4, 5], [1, 0, 4]])
        a = ImplicitALS(factors=3, iterations=3, seed=7).fit(r).score([0, 1, 2])
        b = ImplicitALS(factors=3, iterations=3, seed=7).fit(r).score([0, 1, 2])
        np.testing.assert_array_equal(a, b)

if __name__ == '__main__':
    unittest.main()
