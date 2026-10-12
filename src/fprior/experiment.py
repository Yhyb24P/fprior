"""Shared nested selection for fixed ridge and estimator families.

The caller supplies frozen IDs/folds and a fold-design factory. Outer responses
are not accepted: predictions are completed before evaluation may load them.
"""
from __future__ import annotations

import hashlib
import time
import numpy as np
from .baselines import elasticnet_grid, gbrt_grid, fit_baseline, fit_elasticnet_path
from .io import finite_array
from .solver import simplex_grid, solve_weight, select_configuration


def ridge_candidates(names, config, *, active=None):
    weights = simplex_grid(len(names), config['solver']['simplex_denominator'], active=active)
    return [{'weights': w, 'rho': float(rho), 'config_index': i}
            for i, (w, rho) in enumerate((w, r) for w in weights for r in config['solver']['rho'])]


def equal_budget_weights(config, *, active=None):
    """Fixed hash subsets, always including additive; no outcome-dependent seed."""
    full = simplex_grid(3, config['solver']['simplex_denominator'], active=active)
    count = config['controls']['m2_equal_budget_weights']
    if count < 1 or (active is None and count > len(full)):
        raise ValueError('EQUAL_BUDGET_INVALID')
    count = min(count, len(full))
    additive = (1., 0., 0.) if (1., 0., 0.) in full else full[-1]
    result = []
    for repeat in range(config['controls']['equal_budget_replicates']):
        state = np.random.SeedSequence([config['seed'], 1, repeat]).generate_state(4)
        prefix = ','.join(map(str, state.tolist()))
        def key(weight):
            text = prefix+'\t'+','.join(format(v, '.17g') for v in weight)
            return hashlib.sha256(text.encode()).hexdigest()
        result.append([additive]+sorted([w for w in full if w != additive], key=key)[:count-1])
    return result


def _all_candidates(design, config):
    mandatory = {'M0', 'EN', 'MD', 'MP', 'M1', 'M2', 'MU', 'MW', 'MWG',
                 'MHap', 'MLocal', 'FU', 'FU-GBRT', 'MI', 'MRPerm', 'MI3', 'MRPerm3'}
    if config['controls'].get('sign_permutation_replicates', 0):
        mandatory.add('MSign')
    if set(config['models']['families']) != mandatory:
        raise ValueError('MODEL_COMPARISON_CATALOG_INCOMPLETE')
    families = {name: (list(names), ridge_candidates(names, config,
                active=np.asarray([design.kernels[k][0].any() for k in names])))
                for name, names in design.families.items()}
    active = np.asarray([design.kernels[k][0].any() for k in ('A', 'F', 'R')])
    for repeat, weights in enumerate(equal_budget_weights(config, active=active)):
        families[f'M2Equal:{repeat}'] = (['A', 'F', 'R'], [
            {'weights': w, 'rho': float(r), 'config_index': i}
            for i, (w, r) in enumerate((w, r) for w in weights for r in config['solver']['rho'])])
    families['EN'] = ('X', elasticnet_grid(config['baselines']['elasticnet']))
    families['FU-GBRT'] = ('U', gbrt_grid(config['baselines']['gbrt']))
    return families


def _evaluate_design(design, y, candidates, config, *, selected=None):
    """One eigendecomposition per distinct kernel weight, shared across families."""
    result = {}; cache = {}; diagnostics = {}
    for family, (names, grid) in candidates.items():
        records = []; elapsed = time.perf_counter()
        indices = range(len(grid)) if selected is None else [selected[family]['config_index']]
        if family == 'EN':
            # Outer refits follow the same path through stronger regularization,
            # using outer training responses only. Never warm-start across folds.
            path = grid if selected is None else [p for p in grid
                if p['l1_ratio'] == grid[indices[0]]['l1_ratio']
                and p['alpha_ratio'] >= grid[indices[0]]['alpha_ratio']]
            fits = fit_elasticnet_path(design.features[names][0], y, path,
                                      dict(config['baselines']['elasticnet'],
                                           compute_device=config['execution'].get('compute_device', 'cpu')),
                                      seed=config['seed'])
            by_parameter = {(p['l1_ratio'], p['alpha_ratio']): fit for p, fit in zip(path, fits)}
            for index in indices:
                p = grid[index]; fit = by_parameter[(p['l1_ratio'], p['alpha_ratio'])]
                records.append({'predictions': fit.predict(design.features[names][1]),
                                'diagnostics': fit.diagnostics})
            result[family] = records
            diagnostics[family] = {'seconds': time.perf_counter()-elapsed,
                                  'candidate_count': len(grid), 'path_fits': len(path),
                                  'fits': [fit.diagnostics for fit in fits]}
            continue
        for index in indices:
            parameter = grid[index]
            if family in ('EN', 'FU-GBRT'):
                settings = config['baselines']['elasticnet' if family == 'EN' else 'gbrt']
                fit = fit_baseline(family, design.features[names][0], y, parameter,
                                   settings, seed=config['seed'])
                records.append({'predictions': fit.predict(design.features[names][1]),
                                'diagnostics': fit.diagnostics})
            else:
                # Zero coefficients remove terms from the cache key, sharing KA
                # decompositions across M0 and every additive control endpoint.
                key = tuple((name, w) for name, w in zip(names, parameter['weights']) if w)
                if not key:
                    key = ((names[0], 0.),)
                if key not in cache:
                    cache[key] = solve_weight([design.kernels[name][0] for name, _ in key],
                        [design.kernels[name][1] for name, _ in key],
                        [w for _, w in key], config['solver']['rho'], y,
                        relative_residual_max=config['solver']['relative_residual_max'],
                        spectral_relative_tolerance=config['solver']['spectral_relative_tolerance'])
                record = next(r for r in cache[key] if r['rho'] == parameter['rho'])
                records.append({'predictions': record['predictions'], 'diagnostics': {
                    'relative_residual': record['relative_residual']}})
        result[family] = records
        diagnostics[family] = {'seconds': time.perf_counter()-elapsed, 'candidate_count': len(grid)}
    diagnostics['unique_weight_decompositions'] = len(cache)
    return result, diagnostics


def nested_predictions(train, test, y_train, inner_folds, build, config, *, groups=None, progress=None):
    """Four-fold pooled selection, outer refit, frozen predictions, no outer y.

    build(train_rows, test_rows) must fit QC/centering/normalization only on its
    first argument. Numerical or convergence failures terminate the comparison.
    """
    y = finite_array(y_train, 1, 'outer training labels')
    inner = np.asarray(inner_folds)
    if len(y) != len(train) or inner.shape != y.shape or inner.dtype.kind not in 'iu':
        raise ValueError('INNER_SPLIT_SHAPE_INVALID')
    folds = list(range(config['evaluation']['inner_folds']))
    if set(inner.tolist()) != set(folds):
        raise ValueError('INNER_FOLDS_INCOMPLETE')
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape or any(len(set(inner[groups == g])) != 1 for g in np.unique(groups)):
            raise ValueError('INNER_GROUP_LEAKAGE')
    sums = None; candidates = None; audit = []; count = 0
    for fold in folds:
        training = inner != fold; validation = ~training
        start = time.perf_counter()
        design = build(train[training], train[validation])
        current = _all_candidates(design, config)
        if candidates is None:
            candidates = current
            sums = {name: np.zeros(len(grid)) for name, (_, grid) in current.items()}
        elif current != candidates:
            raise ValueError('CANDIDATE_CATALOG_CHANGED')
        try:
            predictions, timing = _evaluate_design(design, y[training], candidates, config)
        except ValueError as exc:
            raise ValueError(f'INNER_FOLD_FAILED:fold={fold}:{exc}') from exc
        for family, records in predictions.items():
            sums[family] += [np.sum((y[validation]-r['predictions'])**2) for r in records]
        count += int(validation.sum())
        audit.append({'inner_fold': fold, 'train_n': int(training.sum()),
            'validation_n': int(validation.sum()), 'seconds': time.perf_counter()-start,
            'models': timing, 'design': design.diagnostics})
        if progress is not None:
            progress({'inner_fold': fold, 'seconds': audit[-1]['seconds'], 'models': timing})
    selected = {}; inner_scores = {}
    for family, (names, grid) in candidates.items():
        scores = sums[family]/count
        if not np.isfinite(scores).all():
            raise ValueError('INNER_SCORE_NONFINITE')
        inner_scores[family] = scores.tolist()
        if family in ('EN', 'FU-GBRT'):
            best = float(scores.min()); tol = config['solver']['selection_relative_tolerance']
            index = int(np.flatnonzero(scores <= best+tol*max(1., best))[0])
            selected[family] = {'config_index': index, 'mse': float(scores[index]),
                                'parameters': grid[index]}
        else:
            records = [dict(p, mse=float(score), status='PASSED') for p, score in zip(grid, scores)]
            selected[family] = select_configuration(records, tol=config['solver']['selection_relative_tolerance'])
    start = time.perf_counter()
    design = build(train, test)
    outer, timing = _evaluate_design(design, y, candidates, config, selected=selected)
    predictions = {family: records[0]['predictions'] for family, records in outer.items()}
    deployment_order = [name for name in config['models']['families'] if name in ('M0', 'M1', 'M2', 'MU', 'EN')]
    best = min(selected[name]['mse'] for name in deployment_order)
    tolerance = config['solver']['selection_relative_tolerance']*max(1., best)
    deployment = next(name for name in deployment_order if selected[name]['mse'] <= best+tolerance)
    predictions['Deployment'] = predictions[deployment].copy()
    additive_order = [name for name in deployment_order if name in ('M0', 'EN')]
    best_additive = min(selected[name]['mse'] for name in additive_order)
    additive = next(name for name in additive_order if selected[name]['mse'] <=
                    best_additive+config['solver']['selection_relative_tolerance']*max(1., best_additive))
    predictions['AdditiveSelection'] = predictions[additive].copy()
    return {'predictions': predictions, 'selected': selected, 'inner_scores': inner_scores,
            'deployment_family': deployment, 'additive_selection_family': additive,
            'training_mean': float(y.mean()),
            'inner_audit': audit, 'outer_audit': {'seconds': time.perf_counter()-start,
                'models': timing, 'design': design.diagnostics,
                'selected_diagnostics': {f: r[0]['diagnostics'] for f, r in outer.items()}},
            'status': 'PREDICTIONS_FROZEN', 'outer_labels_accessed': False}
