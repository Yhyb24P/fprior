import numpy as np
from numpy.testing import assert_allclose
from fprior.features import boolean_coefficients, local_interaction_counts, permuted_residual_features


def test_two_snp_residual_is_weighted_haplotype_cooccurrence_with_same_mask():
    outputs = np.array([3., 5., 7., 20.])  # 00,01,10,11
    coefficients = boolean_coefficients(outputs)
    assert_allclose(coefficients, [3, 2, 4, 11])
    q = np.array([[0, 0], [2, 2], [1, 2], [1, 1], [np.nan, 2]])
    blocks, mask, subsets = local_interaction_counts(q, [[0, 1]])
    assert subsets == [[3]]
    assert blocks[0][:, 0].tolist() == [0, 2, 1, 0, 0]
    assert mask[:, 0].tolist() == [True, True, True, False, False]
    # Explicit F(h1)+F(h2)-2F(00)-A for each uniquely resolved diploid.
    assert_allclose(blocks[0][:3, 0]*coefficients[3], [0, 22, 11])
    phase = np.array([[[0, 1], [0, 1]], [[0, 1], [1, 0]]], dtype=float)
    phased, valid, _ = local_interaction_counts(np.ones((2, 2)), [[0, 1]], phased_alleles=phase)
    assert valid.all() and phased[0][:, 0].tolist() == [1, 0]


def test_three_snp_cube_keeps_all_pair_terms_and_triple():
    outputs = []
    for mask in range(8):
        a, b, c = [(mask >> bit) & 1 for bit in [2, 1, 0]]
        outputs.append(2+a+2*b+3*c+7*a*b+11*a*c+13*b*c+17*a*b*c)
    coefficient = boolean_coefficients(outputs)
    blocks, valid, subsets = local_interaction_counts(np.array([[2, 2, 2], [1, 2, 0]]), [[0, 1, 2]])
    assert subsets == [[3, 5, 6, 7]]
    assert_allclose(coefficient[subsets[0]], [13, 11, 7, 17])
    assert_allclose(blocks[0]@coefficient[subsets[0]], [96, 7])
    assert valid.all()


def test_teacher_weight_permutation_preserves_arity_and_all_coefficients():
    blocks = [np.array([[0.], [1.]]), np.array([[2.], [0.]]), np.eye(4)[:2]]
    coefficient = [np.array([2.]), np.array([5.]), np.array([3., 7., 11., 13.])]
    values, assignment = permuted_residual_features(blocks, coefficient, np.random.default_rng(3))
    assert set(assignment[:2]) == {0, 1} and assignment[2] == 2
    assert_allclose(values[:, 2], [3, 7])
    repeated, same = permuted_residual_features(blocks, coefficient, np.random.default_rng(3))
    assert_allclose(values, repeated); assert_allclose(assignment, same)
