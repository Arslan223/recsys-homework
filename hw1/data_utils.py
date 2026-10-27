from pathlib import Path
import ast
import gzip
import hashlib
import json
import os
import urllib.request
import numpy as np
import pandas as pd
from scipy import sparse

SOURCE = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon/links_local.html'
BASE = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/'
FILES = ('ratings_Video_Games.csv', 'meta_Video_Games.json.gz')
TRAIN_END = int(pd.Timestamp('2013-01-01', tz='UTC').timestamp())
VALID_END = int(pd.Timestamp('2014-01-01', tz='UTC').timestamp())
MAX_ITEMS = 3000
MIN_TRAIN_REVIEWS = 3


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_data(cache=None):
    cache = Path(cache or os.environ.get('RECSYS_DATA_DIR', Path(__file__).resolve().parents[1] / '.cache/recsys'))
    cache.mkdir(parents=True, exist_ok=True)
    manifest_file = Path(__file__).parent / 'source_checksums.json'
    expected = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    for name in FILES:
        path = cache / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.partial')
            print('Downloading', BASE + name, flush=True)
            urllib.request.urlretrieve(BASE + name, temporary)
            temporary.replace(path)
        if name in expected and sha256(path) != expected[name]:
            raise ValueError(f'Checksum mismatch: {path}; remove the damaged file and rerun')
    return cache


def prepare_data(cache):
    raw = pd.read_csv(cache / FILES[0], names=['reviewerID', 'asin', 'rating', 'timestamp'],
                      dtype={'reviewerID': str, 'asin': str})
    audit = {'raw_rows': len(raw), 'raw_users': raw.reviewerID.nunique(),
             'raw_items': raw.asin.nunique(), 'missing': raw.isna().sum().to_dict(),
             'exact_duplicates': int(raw.duplicated().sum()),
             'repeated_user_item': int(raw.duplicated(['reviewerID', 'asin']).sum())}
    clean = raw.dropna().copy()
    clean = clean[clean.rating.isin([1., 2., 3., 4., 5.]) & (clean.timestamp > 0)].copy()

    clean = clean.sort_values(['timestamp', 'reviewerID', 'asin', 'rating'], kind='stable')
    clean = clean.drop_duplicates(['reviewerID', 'asin'], keep='first')
    clean['timestamp'] = clean.timestamp.astype('int64')
    audit['clean_rows'] = len(clean)
    history = clean[clean.timestamp < TRAIN_END]
    counts = history.groupby('asin').size().rename('count').reset_index()
    catalog = counts.sort_values(['count', 'asin'], ascending=[False, True]).head(MAX_ITEMS).asin
    historical_catalog = history[history.asin.isin(catalog)]
    user_counts = historical_catalog.groupby('reviewerID').size()
    users = np.sort(user_counts[user_counts >= MIN_TRAIN_REVIEWS].index.to_numpy())

    items = np.sort(historical_catalog[historical_catalog.reviewerID.isin(users)].asin.unique())
    selected = clean[clean.reviewerID.isin(users) & clean.asin.isin(items)].copy()
    user_map, item_map = dict(zip(users, range(len(users)))), dict(zip(items, range(len(items))))
    selected['user'] = selected.reviewerID.map(user_map).astype('int32')
    selected['item'] = selected.asin.map(item_map).astype('int32')
    train = selected[selected.timestamp < TRAIN_END].copy()
    valid = selected[(selected.timestamp >= TRAIN_END) & (selected.timestamp < VALID_END)].copy()
    test = selected[selected.timestamp >= VALID_END].copy()
    audit.update(selected_rows=len(selected), users=len(users), items=len(items),
                 train_rows=len(train), validation_rows=len(valid), test_rows=len(test),
                 sparsity=1 - len(selected) / (len(users) * len(items)),
                 first_timestamp=int(selected.timestamp.min()), last_timestamp=int(selected.timestamp.max()))
    exclusions = []
    for label, low, high in [('validation', TRAIN_END, VALID_END), ('test', VALID_END, np.inf)]:
        part = clean[(clean.timestamp >= low) & (clean.timestamp < high)]
        known_user, known_item = part.reviewerID.isin(users), part.asin.isin(items)
        exclusions.append({'split': label, 'all_reviews': len(part),
                           'outside_user_cohort': int((~known_user).sum()),
                           'outside_catalog': int((~known_item).sum()),
                           'excluded_union': int((~(known_user & known_item)).sum()),
                           'retained': int((known_user & known_item).sum())})
    metadata = []
    with gzip.open(cache / FILES[1], 'rt', encoding='utf-8') as stream:
        for line in stream:
            obj = ast.literal_eval(line)
            if obj['asin'] in item_map:
                paths = obj.get('categories', [])
                metadata.append({'asin': obj['asin'], 'title': obj.get('title', ''),
                                 'category': ' / '.join(paths[0]) if paths else '',
                                 'brand': obj.get('brand', '')})
    metadata = pd.DataFrame(metadata).drop_duplicates('asin')
    metadata = pd.DataFrame({'asin': items}).merge(metadata, on='asin', how='left').fillna('')
    assert 100_000 <= len(selected) <= 2_000_000
    assert len(users) >= 1000 and len(items) >= 1000
    assert train.timestamp.max() < valid.timestamp.min() <= valid.timestamp.max() < test.timestamp.min()
    assert not selected.duplicated(['user', 'item']).any()
    return {'selected': selected, 'train': train, 'valid': valid, 'test': test,
            'users': users, 'items': items, 'shape': (len(users), len(items)),
            'metadata': metadata, 'audit': audit, 'exclusions': pd.DataFrame(exclusions)}


def matrices(frame, shape):
    indices = (frame.user.to_numpy(), frame.item.to_numpy())
    rating = sparse.csr_matrix((frame.rating.to_numpy(float), indices), shape=shape)
    timestamp = sparse.csr_matrix((frame.timestamp.to_numpy(float), indices), shape=shape)
    positive = rating.copy()
    positive.data = (positive.data >= 4).astype(float)
    positive.eliminate_zeros()
    seen = rating.copy(); seen.data[:] = 1.
    return rating, timestamp, positive, seen


def build_features(history, shape, cutoff, metadata=True):
    n_users, n_items = shape
    uf, itf = [np.arange(n_users)], [n_users + np.arange(n_items)]
    next_id = n_users + n_items
    if not metadata:
        return np.stack(uf, axis=1), np.stack(itf, axis=1), pd.DataFrame()
    user = history.groupby('user').timestamp.agg(['size', 'min', 'max']).reindex(range(n_users))
    item = history.groupby('item').timestamp.agg(['size', 'min', 'max']).reindex(range(n_items))
    recent = history[history.timestamp >= cutoff - 180 * 86400].groupby('item').size().reindex(range(n_items), fill_value=0)
    specifications = [
        ('user_review_count', user['size'].to_numpy(), [5, 10, 20, 50, 100], uf),
        ('user_days_since_last_review', (cutoff - user['max'].to_numpy()) / 86400, [30, 90, 180, 365, 730], uf),
        ('item_review_count', item['size'].to_numpy(), [10, 30, 100, 300, 1000], itf),
        ('item_days_since_first_review', (cutoff - item['min'].to_numpy()) / 86400, [30, 180, 365, 730, 1825], itf),
        ('item_recent_review_share', recent.to_numpy() / item['size'].to_numpy(), [.05, .2, .5, .8], itf),
    ]
    summaries = []
    for name, values, boundaries, destination in specifications:
        assert np.isfinite(values).all()
        buckets = np.digitize(values, boundaries)
        destination.append(next_id + buckets)
        next_id += len(boundaries) + 1
        summaries.append({'feature': name, 'min': float(values.min()), 'median': float(np.median(values)),
                          'max': float(values.max()), 'bins': str(boundaries)})
    return np.stack(uf, axis=1), np.stack(itf, axis=1), pd.DataFrame(summaries)
