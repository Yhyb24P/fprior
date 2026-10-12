"""Paired development metrics and descriptive group bootstrap; no sealed access."""
from __future__ import annotations
import numpy as np
from .io import finite_array


def _paired(y, prediction):
    y = finite_array(y, 1, 'observed development labels')
    prediction = finite_array(prediction, 1, 'predictions')
    if not len(y) or prediction.shape != y.shape:
        raise ValueError('EVALUATION_SAMPLE_ALIGNMENT_INVALID')
    return y, prediction


def pooled_mse(y, yhat):
    y, prediction = _paired(y, yhat)
    result = float(np.mean((y-prediction)**2))
    if not np.isfinite(result):
        raise ValueError('EVALUATION_ERROR_NONFINITE')
    return result


def q2(y, yhat, train_means):
    y, prediction = _paired(y, yhat)
    _, means = _paired(y, train_means)
    denominator = float(np.sum((y-means)**2))
    return 1-float(np.sum((y-prediction)**2))/denominator if denominator > 0 else None


def paired_metrics(y, predictions, *, train_means=None, reference='M0'):
    if reference not in predictions:
        raise ValueError('REFERENCE_PREDICTIONS_MISSING')
    baseline = pooled_mse(y, predictions[reference])
    records = {}
    for family, prediction in predictions.items():
        observed, predicted = _paired(y, prediction)
        a, b = observed-observed.mean(), predicted-predicted.mean()
        var_a, var_b = float(a@a), float(b@b)
        mse = pooled_mse(observed, predicted)
        slope = float(a@b/var_b) if var_b > 0 else None
        records[family] = dict(mse=mse, mae=float(np.mean(np.abs(observed-predicted))),
            pcc=float(np.clip(a@b/np.sqrt(var_a*var_b), -1, 1)) if min(var_a, var_b) > 0 else None,
            calibration_slope=slope,
            calibration_intercept=float(observed.mean()-slope*predicted.mean()) if slope is not None else None,
            relative_mse_gain_vs_reference=(baseline-mse)/baseline if baseline > 0 else None,
            paired_squared_error_reduction_vs_reference=baseline-mse,
            q2=q2(observed, predicted, train_means) if train_means is not None else None)
    return records


def group_bootstrap_interval(y, yhat_a, yhat_b, groups, replicates, seed):
    """Positive differences favor A; sample groups, retain their original sizes.

    Intervals condition on frozen predictions, not independent folds or a refitted
    selection procedure. Groups have equal sampling probability; material counts
    still determine the pooled error within each bootstrap draw.
    """
    observed, a = _paired(y, yhat_a)
    _, b = _paired(y, yhat_b)
    groups = np.asarray(groups)
    if groups.shape != observed.shape or replicates < 2:
        raise ValueError('BOOTSTRAP_INPUT_INVALID')
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('INSUFFICIENT_BOOTSTRAP_GROUPS')
    differences = (observed-b)**2-(observed-a)**2
    sums = np.bincount(inverse, weights=differences)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = []
    chunk = max(1, 1048576//len(unique))
    for offset in range(0, replicates, chunk):
        draws = rng.integers(len(unique), size=(min(chunk, replicates-offset), len(unique)))
        estimates.extend((sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)).tolist())
    lower, upper = np.quantile(estimates, [.025, .975])
    return dict(paired_squared_error_reduction=float(differences.mean()),
                lower=float(lower), upper=float(upper), groups=len(unique),
                samples=len(observed), replicates=replicates,
                interpretation='descriptive_conditional_on_frozen_predictions')


def continue_decision(relative_gain, outer_signs, *, min_fraction, required_gain,
                      comparisons_complete=False, critical_controls_passed=False,
                      complete_coverage_passed=False):
    if (not np.isfinite(relative_gain) or not outer_signs
            or not set(outer_signs) <= {-1, 0, 1}
            or not 0 < min_fraction <= 1 or not np.isfinite(required_gain) or required_gain < 0):
        raise ValueError('DECISION_INPUT_INVALID')
    if not comparisons_complete:
        return 'COMPARISON_INCOMPLETE'
    if (relative_gain < required_gain
            or sum(sign > 0 for sign in outer_signs)/len(outer_signs) < min_fraction
            or not critical_controls_passed or not complete_coverage_passed):
        return 'INCONCLUSIVE_UNDER_BUDGET'
    return 'ELIGIBLE_FOR_NEXT_STAGE_APPLICATION'


def development_summary(y, predictions, outer_folds, groups, train_means, config):
    """Development comparisons; selection predictions must already be locked."""
    primary = config['development_evaluation']['primary_reference']
    auxiliary = config['development_evaluation']['auxiliary_reference']
    folds = np.asarray(outer_folds); groups = np.asarray(groups)
    if folds.shape != np.asarray(y).shape or groups.shape != folds.shape:
        raise ValueError('EVALUATION_SAMPLE_ALIGNMENT_INVALID')
    if set(folds.tolist()) != set(range(config['evaluation']['outer_folds'])):
        raise ValueError('OUTER_FOLDS_INCOMPLETE')
    if any(len(set(folds[groups == g])) != 1 for g in np.unique(groups)):
        raise ValueError('OUTER_GROUP_LEAKAGE')
    metrics = paired_metrics(y, predictions, train_means=train_means, reference=primary)
    auxiliary_metrics = paired_metrics(y, predictions, train_means=train_means, reference=auxiliary)
    contrasts = [(f, primary) for f in predictions if f != primary]
    contrasts += [(f, auxiliary) for f in predictions if f != auxiliary]
    contrasts += [('M1', 'MD'), ('M1', 'EN'), ('M2', 'EN')]
    contrasts += [('M1', f) for f in predictions if f.startswith(('MP:', 'MSign:'))]
    contrasts += [('M2', f) for f in predictions if f in
        ('M1', 'MU', 'MW', 'MWG', 'MHap', 'MLocal', 'MI', 'MI3')
        or f.startswith(('MRPerm:', 'MRPerm3:'))]
    contrasts = list(dict.fromkeys(contrasts))
    if any(a not in predictions or b not in predictions for a, b in contrasts):
        raise ValueError('COMPARISON_INCOMPLETE')
    paired = {}
    for a, b in contrasts:
        rows = []
        for fold in range(config['evaluation']['outer_folds']):
            mask = folds == fold
            ma = pooled_mse(np.asarray(y)[mask], predictions[a][mask])
            mb = pooled_mse(np.asarray(y)[mask], predictions[b][mask])
            rows.append(dict(outer_fold=fold, samples=int(mask.sum()), mse_model=ma,
                mse_reference=mb, mse_reduction=mb-ma,
                relative_mse_gain=(mb-ma)/mb if mb > 0 else None,
                direction=int(np.sign(mb-ma))))
        ma, mb = metrics[a]['mse'], metrics[b]['mse']
        paired[f'{a}_vs_{b}'] = dict(model=a, reference=b, pooled_mse_model=ma,
            pooled_mse_reference=mb, pooled_mse_reduction=mb-ma,
            relative_pooled_mse_gain=(mb-ma)/mb if mb > 0 else None,
            improving_outer_folds=sum(r['direction'] > 0 for r in rows),
            improving_outer_fraction=sum(r['direction'] > 0 for r in rows)/len(rows),
            outer_folds=rows)
    # Reuse identical whole-group draws for every contrast. Matrix products
    # compute the same weighted sums as separate paired bootstrap calls.
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('INSUFFICIENT_BOOTSTRAP_GROUPS')
    sizes = np.bincount(inverse)
    errors = {f: np.bincount(inverse, weights=(np.asarray(y)-p)**2,
                            minlength=len(unique)) for f, p in predictions.items()}
    deltas = np.stack([errors[b]-errors[a] for a, b in contrasts])
    count = config['evaluation']['descriptive_group_bootstrap_replicates']
    if count < 2:
        raise ValueError('BOOTSTRAP_INPUT_INVALID')
    rng = np.random.default_rng(config['seed']); estimates = []
    for start in range(0, count, 32):
        n = min(32, count-start)
        draws = rng.integers(len(unique), size=(n, len(unique)))
        weights = np.bincount((draws+np.arange(n)[:, None]*len(unique)).ravel(),
                              minlength=n*len(unique)).reshape(n, len(unique))
        estimates.append((deltas@weights.T)/(weights@sizes)[None, :])
    intervals = np.quantile(np.concatenate(estimates, axis=1), [.025, .975], axis=1)
    for i, (a, b) in enumerate(contrasts):
        paired[f'{a}_vs_{b}']['descriptive_group_bootstrap'] = dict(
            lower=float(intervals[0, i]), upper=float(intervals[1, i]),
            replicates=count, groups=len(unique), samples=len(y),
            interpretation='conditional_on_locked_predictions_not_independent_validation')
    return dict(primary_reference=primary, auxiliary_reference=auxiliary,
        metrics=metrics, auxiliary_metrics_vs_M0=auxiliary_metrics,
        paired_comparisons=paired, scope='method_development',
        sealed_evaluation_authorized=False, independent_validation=False)
