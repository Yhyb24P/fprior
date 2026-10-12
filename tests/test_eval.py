import numpy as np
import pytest
from fprior.eval import pooled_mse, q2, paired_metrics, group_bootstrap_interval, continue_decision


def test_metrics_use_fold_training_means_and_constant_cases_are_undefined():
    y = np.array([0., 2., 4.])
    predictions = {'M0': np.ones(3), 'M2': y/2}
    metrics = paired_metrics(y, predictions, train_means=np.array([0., 0., 1.]))
    assert pooled_mse(y, predictions['M0']) == 11/3
    assert q2(y, predictions['M2'], [0, 0, 1]) == pytest.approx(1-5/13)
    assert metrics['M2']['calibration_slope'] == pytest.approx(2)
    assert metrics['M2']['pcc'] == pytest.approx(1)
    assert metrics['M0']['calibration_slope'] is None
    assert metrics['M0']['pcc'] is None
    assert q2([1, 1], [1, 1], [1, 1]) is None
    with pytest.raises(ValueError, match='ALIGNMENT'):
        pooled_mse(y, [0, 1])


def test_bootstrap_resamples_whole_unequal_groups_not_individual_rows():
    y = np.zeros(4); a = np.zeros(4); b = np.array([1., 1., 1., 3.])
    groups = np.array(['large', 'large', 'large', 'small'])
    result = group_bootstrap_interval(y, a, b, groups, 100, 3)
    assert result['paired_squared_error_reduction'] == 3
    rng = np.random.default_rng(3)
    draws = rng.integers(2, size=(100, 2))
    expected = np.array([3., 9.])[draws].sum(axis=1)/np.array([3, 1])[draws].sum(axis=1)
    assert result['lower'] == np.quantile(expected, .025)
    assert result['upper'] == np.quantile(expected, .975)
    assert result == group_bootstrap_interval(y, a, b, groups, 100, 3)


def test_positive_gain_alone_cannot_authorize_next_stage():
    settings = dict(min_fraction=.8, required_gain=.02)
    assert continue_decision(.1, [1]*10, **settings) == 'COMPARISON_INCOMPLETE'
    settings.update(comparisons_complete=True, critical_controls_passed=True,
                    complete_coverage_passed=True)
    assert continue_decision(.03, [1]*8+[-1]*2, **settings) == 'ELIGIBLE_FOR_NEXT_STAGE_APPLICATION'
    assert continue_decision(.03, [1]*7+[-1]*3, **settings) == 'INCONCLUSIVE_UNDER_BUDGET'


def test_development_primary_reference_fold_gains_and_paired_bootstrap():
    from fprior.eval import development_summary
    y = np.zeros(5); folds = np.array([0, 0, 1, 1, 1])
    groups = np.array(['a', 'a', 'b', 'c', 'c'])
    predictions = {'M0': np.full(5, 3.), 'AdditiveSelection': np.ones(5),
                   'M1': np.array([.5, .5, 2., 2., 2.]), 'M2': np.full(5, .8),
                   'EN': np.ones(5), 'MD': np.full(5, 2.)}
    c = dict(seed=31, development_evaluation=dict(primary_reference='AdditiveSelection',
             auxiliary_reference='M0'), evaluation=dict(outer_folds=2,
             descriptive_group_bootstrap_replicates=100))
    result = development_summary(y, predictions, folds, groups, np.ones(5), c)
    record = result['paired_comparisons']['M1_vs_AdditiveSelection']
    assert record['pooled_mse_model'] == pytest.approx(2.5)
    assert record['relative_pooled_mse_gain'] == pytest.approx(-1.5)
    assert record['improving_outer_folds'] == 1
    assert [r['direction'] for r in record['outer_folds']] == [1, -1]
    assert result['auxiliary_metrics_vs_M0']['M1']['relative_mse_gain_vs_reference'] > 0
    expected = group_bootstrap_interval(y, predictions['M1'], predictions['AdditiveSelection'], groups, 100, 31)
    for key in ['lower', 'upper']:
        assert record['descriptive_group_bootstrap'][key] == pytest.approx(expected[key])
    with pytest.raises(ValueError, match='OUTER_GROUP_LEAKAGE'):
        development_summary(y, predictions, folds, ['a']*5, np.ones(5), c)
