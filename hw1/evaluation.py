import numpy as np
import pandas as pd


def relevance(frame):
    return {int(user): np.sort(group.item.unique())
            for user, group in frame[frame.rating >= 4].groupby('user', sort=True)}


def ranking_metrics(recommended, relevant, k=10):
    hit = np.isin(np.asarray(recommended)[:k], relevant)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = discounts[:min(k, len(relevant))].sum()
    return float((hit * discounts).sum() / ideal), float(hit.sum() / len(relevant)), float(hit.any())


def evaluate(model, truth, seen, k=10, batch_size=256):
    users = np.array(sorted(truth))
    rows, exposed = [], set()
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        top = model.recommend(batch, seen, k)
        for user, recommended in zip(batch, top):
            assert len(set(recommended)) == k
            assert not np.intersect1d(recommended, seen[user].indices).size
            assert not np.intersect1d(truth[user], seen[user].indices).size
            ndcg, recall, hit = ranking_metrics(recommended, truth[user], k)
            rows.append((user, ndcg, recall, hit))
            exposed.update(map(int, recommended))
    per_user = pd.DataFrame(rows, columns=['user', 'NDCG@10', 'Recall@10', 'HitRate@10'])
    result = per_user.drop(columns='user').mean().to_dict()
    result['Coverage@10'] = len(exposed) / seen.shape[1]
    result['evaluation_users'] = len(users)
    return result, per_user


def paired_bootstrap(first, second, repetitions=2000, seed=42):
    merged = first[['user', 'NDCG@10']].merge(second[['user', 'NDCG@10']], on='user', validate='one_to_one')
    difference = merged['NDCG@10_x'].to_numpy() - merged['NDCG@10_y'].to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(difference, len(difference), replace=True).mean() for _ in range(repetitions)])
    return {'mean_difference': float(difference.mean()), 'ci_low': float(np.quantile(means, .025)),
            'ci_high': float(np.quantile(means, .975)), 'users': len(difference)}
