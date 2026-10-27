from pathlib import Path
from time import perf_counter
import json
import numpy as np
import pandas as pd
from models import ItemKNN, ImplicitALS, EASE, FactorizationMachine, CustomTargetALS
from data_utils import matrices, build_features, TRAIN_END, VALID_END
from evaluation import relevance, evaluate, paired_bootstrap

SEED = 42
SEARCH = {
    'ItemKNN': [dict(neighbors=k, shrinkage=s) for k in [50, 150] for s in [10, 50]],
    'iALS': [dict(factors=f, regularization=r, alpha=20., iterations=12, seed=SEED)
             for f in [32, 64] for r in [10., 50.]],
    'EASE': [dict(regularization=r) for r in [50., 200., 1000.]],
    'FM': [dict(factors=32, epochs=e, learning_rate=lr, weight_decay=1e-5,
                negatives=3, batch_size=4096, seed=SEED) for e in [8, 16] for lr in [.003, .01]],
}


def fit_model(name, params, training, shape, cutoff, with_metadata=True):
    ratings, timestamps, positive, seen = matrices(training, shape)
    start = perf_counter()
    if name == 'ItemKNN':
        model = ItemKNN(**params).fit(positive)
    elif name == 'iALS':
        model = ImplicitALS(**params).fit(ratings)
    elif name == 'EASE':
        model = EASE(**params).fit(positive)
    elif name in ['FM', 'FM_ID_only']:
        user_features, item_features, _ = build_features(training, shape, cutoff, metadata=with_metadata)
        model = FactorizationMachine(**params).fit(ratings, user_features, item_features)
    elif name == 'CustomALS':
        model = CustomTargetALS(**params).fit(ratings, timestamps, cutoff)
    else:
        raise ValueError(name)
    return model, seen, perf_counter() - start


def tune(data, output=Path('results')):
    output.mkdir(exist_ok=True, parents=True)
    truth = relevance(data['valid'])
    records, selected = [], {}
    spaces = dict(SEARCH)
    for name in ['ItemKNN', 'iALS', 'EASE', 'FM', 'CustomALS']:
        if name == 'CustomALS':

            spaces[name] = [dict(selected['iALS'], half_life_days=t) for t in [180., 730.]]
        best = -np.inf
        for params in spaces[name]:
            model, seen, elapsed = fit_model(name, params, data['train'], data['shape'], TRAIN_END)
            metrics, _ = evaluate(model, truth, seen)
            records.append(dict(model=name, parameters=json.dumps(params, sort_keys=True),
                                fit_seconds=elapsed, **metrics))
            if metrics['NDCG@10'] > best:
                best = metrics['NDCG@10']; selected[name] = params.copy()
            del model
        pd.DataFrame(records).to_csv(output / 'validation.csv', index=False)

    (output / 'selected_parameters.json').write_text(json.dumps(selected, indent=2) + '\n')
    (output / 'search_space.json').write_text(json.dumps(spaces, indent=2) + '\n')
    return pd.DataFrame(records), selected


def final_evaluation(data, selected, output=Path('results')):
    training = pd.concat([data['train'], data['valid']], ignore_index=True)
    truth = relevance(data['test'])
    records, per_user, loss_curves = [], {}, {}
    for name in ['ItemKNN', 'iALS', 'EASE', 'FM', 'CustomALS', 'FM_ID_only']:
        params = selected['FM' if name == 'FM_ID_only' else name]
        model, seen, elapsed = fit_model(name, params, training, data['shape'], VALID_END,
                                         with_metadata=name != 'FM_ID_only')
        evaluation_start = perf_counter()
        metrics, per_user[name] = evaluate(model, truth, seen)
        records.append(dict(model=name, parameters=json.dumps(params, sort_keys=True),
                            fit_seconds=elapsed, evaluation_seconds=perf_counter() - evaluation_start, **metrics))
        if name in ['FM', 'FM_ID_only']:
            loss_curves[name] = model.loss_history
        del model
    result = pd.DataFrame(records)
    result.to_csv(output / 'test.csv', index=False)
    comparisons = {
        'FM_minus_ID_only': paired_bootstrap(per_user['FM'], per_user['FM_ID_only']),
        'CustomALS_minus_iALS': paired_bootstrap(per_user['CustomALS'], per_user['iALS']),
    }
    (output / 'paired_bootstrap.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    (output / 'fm_loss.json').write_text(json.dumps(loss_curves, indent=2) + '\n')
    return result, comparisons, loss_curves
