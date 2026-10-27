import numpy as np
import torch
from .common import Recommender


class _FM(torch.nn.Module):
    def __init__(self, n_features, factors):
        super().__init__()
        self.linear = torch.nn.Embedding(n_features, 1)
        self.factors = torch.nn.Embedding(n_features, factors)
        self.bias = torch.nn.Parameter(torch.zeros(()))
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.factors.weight, std=.03)

    def forward(self, features):
        embedding = self.factors(features)
        interactions = .5 * (embedding.sum(1).square() - embedding.square().sum(1)).sum(1)
        return self.bias + self.linear(features).sum((1, 2)) + interactions

    def aggregate(self, features):
        embedding = self.factors(features)
        summed = embedding.sum(1)
        within = .5 * (summed.square() - embedding.square().sum(1)).sum(1)
        return summed, within + self.linear(features).sum((1, 2))


class FactorizationMachine(Recommender):
    def __init__(self, factors=32, epochs=8, learning_rate=.01, weight_decay=1e-5,
                 negatives=3, batch_size=4096, seed=42):
        self.factors, self.epochs, self.learning_rate = factors, epochs, learning_rate
        self.weight_decay, self.negatives = weight_decay, negatives
        self.batch_size, self.seed = batch_size, seed

    def fit(self, ratings, user_features, item_features):
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(self.seed)
        self.uf = torch.as_tensor(user_features, dtype=torch.long)
        self.itf = torch.as_tensor(item_features, dtype=torch.long)
        n_features = int(max(user_features.max(), item_features.max()) + 1)
        net = _FM(n_features, self.factors)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        coo = ratings.tocoo()
        u, i = coo.row.astype(np.int64), coo.col.astype(np.int64)
        labels = (coo.data >= 4).astype(np.float32)
        positive_users = u[labels == 1]
        seen = np.zeros(ratings.shape, dtype=bool)
        seen[u, i] = True
        self.loss_history = []
        for _ in range(self.epochs):
            nu = np.repeat(positive_users, self.negatives)
            ni = rng.integers(ratings.shape[1], size=len(nu))
            bad = seen[nu, ni]
            while np.any(bad):
                ni[bad] = rng.integers(ratings.shape[1], size=int(bad.sum()))
                bad = seen[nu, ni]
            eu, ei = np.concatenate([u, nu]), np.concatenate([i, ni])
            ey = np.concatenate([labels, np.zeros(len(nu), dtype=np.float32)])
            order = rng.permutation(len(ey))
            total_loss = 0.
            for start in range(0, len(order), self.batch_size):
                ix = order[start:start + self.batch_size]
                features = torch.cat([self.uf[eu[ix]], self.itf[ei[ix]]], dim=1)
                logits = net(features)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(ey[ix]))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item() * len(ix)
            self.loss_history.append(total_loss / len(ey))

        with torch.no_grad():
            up, ub = net.aggregate(self.uf)
            ip, ib = net.aggregate(self.itf)
            self.user_vectors, self.user_bias = up.numpy(), ub.numpy()
            self.item_vectors, self.item_bias = ip.numpy(), ib.numpy()
            self.global_bias = float(net.bias)
        self.net = net
        return self

    def score(self, users):
        return (self.user_vectors[users] @ self.item_vectors.T + self.user_bias[users, None]
                + self.item_bias[None, :] + self.global_bias)
