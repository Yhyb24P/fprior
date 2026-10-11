"""Synthetic mathematical contracts; never accesses source genotype/phenotype files."""
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from preprocessing.dosages import DosageTransform
from fprior.features import (WindowVariant, resolve_haplotypes, build_d_matrix,
                            build_features, center_trainwise, verify_identities)
from fprior.kernels import (trace_normalize, cross_kernel, rbf_kernel,
                           diagonal_annotation_kernel, haplotype_kernel, local_rbf_kernels)
from fprior.solver import simplex_grid, solve_weight, select_configuration, predict
from fprior.io import validate_inputs, assert_development_only, assert_order, load_config


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / 'configs/funcprior.toml')[0]


@pytest.fixture
def variants():
    return [WindowVariant(0, 0, 'A', 'C'), WindowVariant(1, 1, 'A', 'C')]


class ToyTeacher:
    def __init__(self, interaction):
        self.interaction = interaction

    def score(self, sequences):
        return np.array([3 + (s[0] == 'C') + 2 * (s[1] == 'C')
                         + self.interaction * (s[:2] == 'CC') for s in sequences])


def test_training_qc_imputation_and_monomorphic_test():
    train = [[0, 0, np.nan, np.nan], [2, 0, 2, np.nan]]
    transform = DosageTransform.fit(train, min_call_rate=.5, maf_min=0)
    assert_array_equal(transform.retained, [True, True, True, False])
    q, x = transform.transform([[np.nan, 2, np.nan, 1]])
    assert_allclose(q, [[1, 2, 2]])
    assert_allclose(x, [[0, 2, 0]])
    assert_allclose(transform.maf, [.5, 0, 0, 0])
    strict = DosageTransform.fit(train, min_call_rate=.98, maf_min=0)
    assert_array_equal(strict.retained, [True, True, False, False])


@pytest.mark.parametrize('bad', [3, np.inf, -.1, 1.5])
def test_raw_encoding_not_silently_converted(bad):
    with pytest.raises(ValueError, match='DOSAGE_INVALID'):
        DosageTransform.fit([[bad]], min_call_rate=0, maf_min=0)


@pytest.mark.parametrize('interaction', [0, 4])
def test_exact_decomposition_and_centering(variants, config, interaction):
    teacher = ToyTeacher(interaction)
    d = build_d_matrix(teacher, ['AAAA'], [variants], n_variants=2)
    assert_allclose(d, [[1], [2]])
    q = np.array([[0, 0], [1, 0], [2, 2], [1, 1]], float)
    outputs = []
    valid = []
    for row in q:
        haplotypes = resolve_haplotypes(row, 'AAAA', variants)
        valid.append([haplotypes is not None])
        outputs.append([np.nan if haplotypes is None else teacher.score(haplotypes).sum() - 6])
    a, u, r, mask = build_features(q, d, outputs, np.asarray(valid))
    assert r[-1, 0] == 0
    assert u[-1, 0] == a[-1, 0]
    if interaction == 0:
        assert_allclose(r, 0)
    else:
        assert r[2, 0] == 8
    train = np.array([0, 2])
    x = center_trainwise(q, train)
    z = center_trainwise(a, train)
    uc = center_trainwise(u, train)
    rc = center_trainwise(r, train)
    assert verify_identities(x, d, z, uc, rc, relative_tolerance=config['solver']['feature_identity_relative_tolerance']) < 1e-12
    if interaction:
        assert rc[-1, 0] == -4  # invalid raw R=0 is not forced back to zero after centering
    assert not mask[-1, 0]


def test_cis_trans_phase_and_order_invariance(variants):
    teacher = ToyTeacher(4)
    cis = resolve_haplotypes([1, 1], 'AAAA', variants, phased_alleles=[[1, 0], [1, 0]])
    trans = resolve_haplotypes([1, 1], 'AAAA', variants, phased_alleles=[[1, 0], [0, 1]])
    assert teacher.score(cis).sum() - teacher.score(trans).sum() == 4
    swapped = resolve_haplotypes([1, 1], 'AAAA', variants, phased_alleles=[[0, 1], [0, 1]])
    assert_allclose(teacher.score(cis).sum(), teacher.score(swapped).sum())
    assert resolve_haplotypes([1, 1], 'AAAA', variants) is None
    assert resolve_haplotypes([np.nan, 0], 'AAAA', variants) is None
    assert resolve_haplotypes([2, 2], 'AAAA', variants) == ('CCAA', 'CCAA')
    with pytest.raises(ValueError, match='PHASE_DOSAGE_MISMATCH'):
        resolve_haplotypes([1, 1], 'AAAA', variants, phased_alleles=[[1, 1], [1, 0]])
    with pytest.raises(ValueError, match='REFERENCE_ALLELE_MISMATCH'):
        resolve_haplotypes([0, 0], 'CAAA', variants)


def test_signed_window_specific_d():
    class SignedTeacher:
        def score(self, sequences):
            return np.array([(-2 if s[-1] == 'A' else 5) * (s[0] == 'C') for s in sequences])
    d = build_d_matrix(SignedTeacher(), ['AAAA', 'AAAC'],
                       [[WindowVariant(0, 0, 'A', 'C')]] * 2, n_variants=2)
    assert_allclose(d, [[-2, 5], [0, 0]])


def test_linear_kernels_trace_and_prior_equivalence():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(9, 6)); x -= x.mean(axis=0)
    d = rng.normal(size=(6, 3))
    z = x @ d
    kernel, c = trace_normalize(z)
    assert_allclose(np.trace(kernel), len(x))
    assert_allclose(kernel * c, x @ (d @ d.T) @ x.T, atol=1e-12)
    v = np.sum(d * d, axis=1)
    raw = x @ np.diag(v) @ x.T
    assert_allclose(diagonal_annotation_kernel(x, v), raw / (np.trace(raw) / len(x)))
    assert_allclose(cross_kernel(z[:2], z, c), kernel[:2])
    zero, scale = trace_normalize(np.zeros((9, 2)))
    assert scale == 0
    assert_allclose(zero, 0)
    assert_allclose(cross_kernel(z[:2], z, 0), 0)


def test_rbf_training_statistics_and_test_batch_invariance():
    train = np.array([[0, 0], [1, 0], [3, 0]], float)
    test = np.array([[2, 0], [100, 0]], float)
    kernel, bandwidth = rbf_kernel(train)
    assert bandwidth == 4
    assert_allclose(kernel.sum(axis=0), 0, atol=1e-14)
    assert_allclose(np.trace(kernel), 3)
    cross, test_bandwidth = rbf_kernel(train, test)
    assert test_bandwidth == bandwidth
    assert_allclose(cross[:1], rbf_kernel(train, test[:1])[0])
    assert_allclose(rbf_kernel(train + 1e8, test + 1e8)[0], cross)
    assert_allclose(rbf_kernel(train, train)[0], kernel)
    assert rbf_kernel(np.ones((3, 2)))[1] == 0
    local, local_cross, removed = local_rbf_kernels([train, np.ones((3, 2))], [test, np.ones((2, 2))])
    assert removed == [1]
    assert_allclose(local, kernel)
    assert_allclose(local_cross, cross)


def test_haplotype_training_categories_and_invalid_mean():
    train = np.array([[2, 0, 0], [0, 2, 0], [np.nan, np.nan, np.nan]])
    test = np.array([[np.nan, np.nan, np.nan], [0, 0, 2]])
    kernel, cross = haplotype_kernel(train, test)
    assert_allclose(np.trace(kernel), 3)
    assert_allclose(cross, 0, atol=1e-14)
    assert_allclose(kernel[-1], 0)


@pytest.mark.parametrize('n,count', [(1, 1), (2, 5), (3, 15)])
def test_simplex_counts(n, count):
    grid = simplex_grid(n, 4)
    assert len(grid) == count
    assert_allclose(np.sum(grid, axis=1), 1)
    assert len(simplex_grid(n, 4, active=[True] + [False] * (n - 1))) == 1
    assert simplex_grid(n, 4, active=[False] * n) == [tuple([0.] * n)]


def test_multi_kernel_primal_dual_and_single_eigendecomposition(monkeypatch, config):
    rng = np.random.default_rng(11)
    features = [rng.normal(size=(12, width)) for width in (4, 3)]
    tests = [rng.normal(size=(5, width)) for width in (4, 3)]
    normalized = []
    kernels, crosses = [], []
    for train, test in zip(features, tests):
        test = test - train.mean(axis=0); train = train - train.mean(axis=0)
        kernel, c = trace_normalize(train)
        normalized.append((train / np.sqrt(c), test / np.sqrt(c)))
        kernels.append(kernel); crosses.append(cross_kernel(test, train, c))
    original = np.linalg.eigh
    calls = []
    def counted(matrix):
        calls.append(1)
        return original(matrix)
    monkeypatch.setattr(np.linalg, 'eigh', counted)
    y = rng.normal(size=12) + 7
    weights = (.25, .75)
    results = solve_weight(kernels, crosses, weights, [.01, .1, 1], y,
                           relative_residual_max=config['solver']['relative_residual_max'],
                           spectral_relative_tolerance=config['solver']['spectral_relative_tolerance'])
    assert len(calls) == 1
    b = np.concatenate([np.sqrt(w) * pair[0] for w, pair in zip(weights, normalized)], axis=1)
    t = np.concatenate([np.sqrt(w) * pair[1] for w, pair in zip(weights, normalized)], axis=1)
    for result in results:
        beta = np.linalg.solve(b.T @ b + len(y) * result['rho'] * np.eye(b.shape[1]), b.T @ (y - y.mean()))
        assert_allclose(result['predictions'], y.mean() + t @ beta, atol=1e-10)
        assert_allclose(predict(result['alpha'], crosses, weights, result['y_mean']), result['predictions'])
        assert result['relative_residual'] < 1e-10


def test_zero_kernel_and_constant_target(config):
    result = solve_weight([np.zeros((3, 3))], [np.zeros((2, 3))], (0,), [.1], np.ones(3) * 4,
                          relative_residual_max=config['solver']['relative_residual_max'],
                          spectral_relative_tolerance=config['solver']['spectral_relative_tolerance'])[0]
    assert_allclose(result['predictions'], 4)
    assert result['relative_residual'] == 0


@pytest.mark.parametrize('kernel', [np.array([[1., 0], [0, 1]]), np.array([[-1., 1], [1, -1]]), np.array([[1., 2], [0, 1]])])
def test_invalid_kernel_fails(kernel, config):
    with pytest.raises(ValueError, match='KERNEL_'):
        solve_weight([kernel], [np.zeros((1, 2))], (1,), [.1], [0, 1],
                     relative_residual_max=config['solver']['relative_residual_max'],
                     spectral_relative_tolerance=config['solver']['spectral_relative_tolerance'])


def test_selection_ties_and_incomplete_comparison(config):
    scores = [dict(status='PASSED', weights=w, rho=rho, mse=1 + delta, config_index=i)
              for i, (w, rho, delta) in enumerate([((.5, .5), 1, 0), ((1., 0), .1, 1e-12), ((1., 0), 1, 1e-12), ((0., 1), 10, 0)])]
    assert select_configuration(scores, tol=config['solver']['selection_relative_tolerance'])['config_index'] == 2
    scores.append(dict(status='FAILED'))
    with pytest.raises(ValueError, match='COMPARISON_INCOMPLETE'):
        select_configuration(scores, tol=config['solver']['selection_relative_tolerance'])


def test_preflight_never_opens_inputs(monkeypatch, tmp_path, config):
    def forbidden(*args, **kwargs):
        raise AssertionError('Input content must not be read by preflight')
    monkeypatch.setattr(Path, 'open', forbidden)
    result = validate_inputs(config, tmp_path)
    assert result['status'] == 'MISSING_REQUIRED_INPUTS'
    assert not result['phenotypes_accessed'] and not result['data_ready']
    assert validate_inputs({'primary_trait_id': ''}, tmp_path)['status'] == 'TARGET_UNSPECIFIED'


def test_sealed_ids_and_order_contract():
    assert_development_only(['a', 'b'], frozenset({'c'}))
    with pytest.raises(ValueError, match='SEALED_ID_ACCESS_FORBIDDEN'):
        assert_development_only(['a', 'c'], frozenset({'c'}))
    with pytest.raises(ValueError, match='SAMPLE_ID_DUPLICATE'):
        assert_development_only(['a', 'a'], frozenset())
    with pytest.raises(ValueError, match='ORDER_MISMATCH'):
        assert_order(['b', 'a'], ['a', 'b'], 'SAMPLE')


def test_gram_rbf_matches_direct_distances_with_duplicate_and_constant_columns():
    from fprior.kernels import GaussianKernel
    rng = np.random.default_rng(19)
    train = rng.normal(size=(9, 41346))
    train[1] = train[0]
    train[:, -4:] = 12
    test = np.vstack([train[1], train[3]+1e-10, rng.normal(size=(1, 41346))])
    distance = np.sum((train[:, None, :]-train[None, :, :])**2, axis=2)
    bandwidth = np.median(distance[np.triu_indices(9, 1)][distance[np.triu_indices(9, 1)] > 0])
    raw = np.exp(-distance/(2*bandwidth))
    centered = raw-raw.mean(axis=1, keepdims=True)-raw.mean(axis=0)+raw.mean()
    scale = np.trace(centered)/len(train)
    direct_cross = np.exp(-np.sum((test[:, None, :]-train[None, :, :])**2, axis=2)/(2*bandwidth))
    expected = (direct_cross-direct_cross.mean(axis=1, keepdims=True)-raw.mean(axis=0)+raw.mean())/scale
    fitted = GaussianKernel.fit(train, raw_gram=train@train.T)
    assert_allclose(fitted.bandwidth, bandwidth, rtol=1e-13)
    assert_allclose(fitted.normalized_train, centered/scale, atol=1e-13)
    assert_allclose(fitted.transform(test, raw_cross=test@train.T), expected, atol=1e-13)


def test_gram_distance_repairs_close_far_offset_pairs():
    from fprior.kernels import _squared_distances
    values = np.array([[0., 0.], [1e10, 1e10], [1e10+.01, 1e10-.02]])
    expected = np.sum((values[:, None, :]-values[None, :, :])**2, axis=2)
    assert_allclose(_squared_distances(values, values), expected, rtol=1e-14, atol=0)
    assert_allclose(_squared_distances(values, values, values@values.T), expected, rtol=1e-14, atol=0)
