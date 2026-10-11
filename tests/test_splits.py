import numpy as np
import pytest

from preprocessing.splits import near_duplicate_groups, plan_partitions


def test_transitive_groups_and_shared_call_boundary():
    values = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0],
                       [2, 2, 2, np.nan]], dtype=float)
    groups, audit = near_duplicate_groups(values, ['a', 'b', 'c', 'd'], .75, .75,
                                         block_size=2)
    assert groups[0] == groups[1] == groups[2] != groups[3]
    assert audit['pair_edges'] == 2
    assert audit['groups'] == 2
    assert not audit['phenotypes_accessed']
    reordered, _ = near_duplicate_groups(values[::-1], ['d', 'c', 'b', 'a'], .75, .75)
    assert reordered[::-1] == groups


def test_missingness_is_not_agreement_and_unknown_pairs_are_conservative():
    values = np.array([[0, 1, np.nan, np.nan], [0, 2, np.nan, np.nan]])
    groups, _ = near_duplicate_groups(values, ['a', 'b'], .5, .75)
    assert groups[0] != groups[1]
    groups, audit = near_duplicate_groups(values, ['a', 'b'], .75, .75)
    assert groups[0] == groups[1]
    assert audit['low_coverage_conservative_edges'] == 1
    values[1, :2] = [2, 2]
    groups, _ = near_duplicate_groups(values, ['a', 'b'], .75, .75)
    assert groups[0] != groups[1]


@pytest.mark.skipif(__import__('torch').cuda.is_available() is False,
                    reason='CUDA is required for the production grouping path')
def test_cuda_counts_exceed_half_precision_exact_integer_range():
    rng = np.random.default_rng(7)
    values = rng.integers(0, 3, size=(5, 41346)).astype(float)
    values[1] = values[0]
    values[2] = values[0]
    values[2, :206] = (values[2, :206] + 1) % 3
    values[3] = values[0]
    values[3, 206:413] = (values[3, 206:413] + 1) % 3
    values[4, :800] = np.nan
    ids = ['a', 'b', 'c', 'd', 'e']
    cpu, audit = near_duplicate_groups(values, ids, .98, .995, block_size=2)
    gpu, gpu_audit = near_duplicate_groups(values, ids, .98, .995,
                                            device='cuda', block_size=2)
    assert cpu == gpu
    assert cpu[0] == cpu[1] == cpu[2] != cpu[3]
    assert audit['pair_edges'] == gpu_audit['pair_edges']


def test_partitions_preserve_groups_sealing_and_nested_training_budgets():
    ids = [f's{i}' for i in range(100)]
    groups = [f'g{i//2}' for i in range(100)]
    partitions, nested, audit = plan_partitions(ids, groups, .2, 5, 2, [12, 24], 7)
    membership = {r['accession_id']: r for r in partitions}
    assert sum(r['partition'] == 'sealed' for r in partitions) == 20
    for group in set(groups):
        records = [r for r in partitions if r['group_id'] == group]
        assert len({(r['partition'], r['outer_fold']) for r in records}) == 1
    for fold in range(5):
        selections = []
        for budget in [12, 24]:
            rows = [r for r in nested if r['outer_fold'] == fold and r['budget'] == budget]
            selections.append({r['accession_id'] for r in rows})
            assert len(rows) <= budget
            assert {r['inner_fold'] for r in rows} == {0, 1}
            for r in rows:
                assert membership[r['accession_id']]['partition'] == 'development'
                assert membership[r['accession_id']]['outer_fold'] != fold
        assert selections[0] < selections[1]
    assert all(r['samples'] <= r['budget'] for r in audit)


def test_impossible_group_budget_fails_without_splitting_or_skipping():
    with pytest.raises(ValueError, match='INSUFFICIENT_GROUPS'):
        plan_partitions([f's{i}' for i in range(20)], [f'g{i//2}' for i in range(20)],
                        .2, 2, 2, [1], 7)
    with pytest.raises(ValueError, match='INSUFFICIENT_GROUPS_FOR_SEALED'):
        plan_partitions(['a', 'b'], ['same', 'same'], .2, 2, 2, [10], 7)
