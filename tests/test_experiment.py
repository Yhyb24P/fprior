"""Synthetic integration checks for shared fold design and nested selection."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from fprior.io import load_config
from fprior.design import build_design
from fprior.experiment import nested_predictions, equal_budget_weights
from fprior.experiment import _all_candidates
from fprior.runner import _acceptance


def fixture():
    config, _ = load_config(Path(__file__).parents[1]/'configs/funcprior.toml')
    config = deepcopy(config)
    config['regions']['min_usable_windows'] = 2
    config['controls'].update(permutation_replicates=1, residual_weight_permutation_replicates=1,
                              equal_budget_replicates=1, m2_equal_budget_weights=2)
    config['solver'].update(simplex_denominator=2, rho=[.01, .1])
    config['baselines']['elasticnet'].update(l1_ratio=[.5], alpha_max_ratios=[1., .1])
    config['baselines']['gbrt'].update(max_depth=[1], learning_rate=[.1], n_estimators=[10], min_samples_leaf=2)
    rng = np.random.default_rng(31)
    q = rng.integers(0, 2, (64, 8)).astype(float)*2
    windows = [[0, 1], [2, 3, 4]]
    cubes = [np.array([0., .2, .4, 1.1]), np.array([0., .1, .3, 1., -.2, .5, .8, 2.])]
    return config, q, windows, cubes


def test_fold_design_train_only_and_complete_interactions():
    c, q, windows, cubes = fixture()
    q[0, 0] = np.nan
    # Permit this missing call in training for this particular fixture.
    c['variant_policy']['min_train_call_rate'] = .95
    a = build_design(q[:48], q[48:], windows, cubes, c, noise_threshold=1e-8)
    changed = q[48:].copy(); changed[:, 0] = 2
    b = build_design(q[:48], changed, windows, cubes, c, noise_threshold=1e-8)
    assert a.features['I'][0].shape[1] == 5  # one pair + three pairs + one triple
    for name in a.kernels:
        np.testing.assert_array_equal(a.kernels[name][0], b.kernels[name][0])
        np.testing.assert_allclose(a.kernels[name][0].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(a.features['U'][0], a.features['F'][0]+a.features['R'][0])
    assert a.diagnostics['valid_fraction'][0] < 1
    np.testing.assert_allclose(np.trace(a.kernels['Local'][0])/48, 1.)


def test_nested_all_families_frozen_outer_predictions_and_groups():
    c, q, windows, cubes = fixture()
    train, test = q[:48], q[48:]
    y = train[:, 0]-.5*train[:, 2]+.2*train[:, 3]*train[:, 4]
    folds = np.arange(48) % 4
    build = lambda a, b: build_design(a, b, windows, cubes, c, noise_threshold=1e-8)
    result = nested_predictions(train, test, y, folds, build, c, groups=np.arange(48).astype(str))
    mandatory = set(c['models']['families'])-{'MP', 'MRPerm', 'MRPerm3'}
    assert mandatory <= set(result['predictions'])
    assert {'MP:0', 'MRPerm:0', 'MRPerm3:0', 'M2Equal:0', 'Deployment'} <= set(result['predictions'])
    assert result['outer_labels_accessed'] is False
    assert result['status'] == 'PREDICTIONS_FROZEN'
    assert all(p.shape == (16,) and np.isfinite(p).all() for p in result['predictions'].values())
    deployment = result['deployment_family']
    np.testing.assert_array_equal(result['predictions']['Deployment'], result['predictions'][deployment])
    group = np.arange(48).astype(str); group[1] = group[0]
    with pytest.raises(ValueError, match='INNER_GROUP_LEAKAGE'):
        nested_predictions(train, test, y, folds, build, c, groups=group)
    for weights in equal_budget_weights(c):
        assert len(weights) == 2 and (1., 0., 0.) in weights


def test_real_evaluation_fails_before_labels_without_pretraining_evidence(tmp_path):
    c, _, _, _ = fixture()
    with pytest.raises(ValueError, match='PRETRAINING_ACCEPTANCE_MISSING'):
        _acceptance(c, tmp_path, None)


def test_matched_residual_controls_share_first_two_kernels_and_search():
    c, q, windows, cubes = fixture()
    design = build_design(q[:48], q[48:], windows, cubes, c, noise_threshold=1e-8)
    catalog = _all_candidates(design, c)
    assert catalog['MI3'][0] == ['A', 'F', 'I']
    assert catalog['MRPerm3:0'][0] == ['A', 'F', 'RP:0']
    assert catalog['MRPerm:0'][0][-1] == catalog['MRPerm3:0'][0][-1]
    assert catalog['MI3'][1] == catalog['M2'][1] == catalog['MRPerm3:0'][1]


def test_linear_teacher_turns_off_residual_without_stopping_additive_comparison():
    c, q, windows, _ = fixture()
    cubes = [np.asarray([sum((i >> bit & 1)*(.1+.2*bit) for bit in range(len(w)))
                         for i in range(1 << len(w))]) for w in windows]
    design = build_design(q[:48], q[48:], windows, cubes, c, noise_threshold=1e-8)
    assert not design.kernels['R'][0].any()
    assert design.kernels['F'][0].any()
    candidates = _all_candidates(design, c)
    assert all(p['weights'][2] == 0 for p in candidates['M2'][1])
    assert len(candidates['M2'][1]) == 6  # three two-kernel weights times two rhos
