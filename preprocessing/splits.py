"""Phenotype-free, exact-call near-duplicate groups for the official cohort."""
from fractions import Fraction
import hashlib
from collections import Counter

import numpy as np


def near_duplicate_groups(dosages, sample_ids, min_shared_call_fraction,
                          min_concordance, *, device='cpu', block_size=512):
    """Connect pairs with sufficient shared calls and exact dosage concordance.

    Uses all catalog variants, without frequency filtering or imputation. CUDA
    uses binary float16 inputs and float32 accumulation; integer counts below
    2**24 are exactly representable. Missing calls never contribute agreement.
    Low-coverage pairs connect conservatively unless observed discordances alone
    exclude full-catalog near duplication. No missing calls are imputed. Connected
    components protect against transitive leakage across partitions.
    """
    import torch

    values = np.asarray(dosages)
    if values.ndim != 2 or len(sample_ids) != len(values):
        raise ValueError('GROUP_SAMPLE_AXIS_INVALID')
    if len(set(sample_ids)) != len(sample_ids) or not all(sample_ids):
        raise ValueError('GROUP_SAMPLE_ID_INVALID')
    n, variants = values.shape
    if not 0 < variants < 2**24 or block_size <= 0:
        raise ValueError('GROUP_COUNT_RANGE_INVALID')
    shared = Fraction(str(min_shared_call_fraction))
    concordance = Fraction(str(min_concordance))
    if not 0 < shared <= 1 or not 0 < concordance <= 1:
        raise ValueError('GROUP_THRESHOLD_INVALID')
    called = np.isfinite(values)
    if np.isinf(values).any() or not np.isin(values[called], [0, 1, 2]).all():
        raise ValueError('GROUP_DOSAGE_INVALID')
    counts = called.sum(axis=1)
    target = torch.device(device)
    dtype = torch.float16 if target.type == 'cuda' else torch.float32
    indicators = [torch.as_tensor(values == allele, dtype=dtype, device=target)
                  for allele in (0, 1, 2)]
    indicators.append(torch.as_tensor(called, dtype=dtype, device=target))
    del called
    parent = list(range(n))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edges = conservative_edges = 0
    with torch.inference_mode():
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            matrices = []
            for indicator in indicators:
                kwargs = {'out_dtype': torch.float32} if target.type == 'cuda' else {}
                matrices.append(torch.mm(indicator[start:end], indicator.T, **kwargs))
            agreement = matrices[0] + matrices[1] + matrices[2]
            common = matrices[3]
            # Integer arithmetic avoids rounding at declared threshold boundaries.
            common = common.to(torch.int64)
            agreement = agreement.to(torch.int64)
            sufficient = common * shared.denominator >= variants * shared.numerator
            unresolved = (~sufficient & ((common - agreement) * concordance.denominator
                          <= variants * (concordance.denominator - concordance.numerator)))
            qualifying = ((sufficient & (agreement * concordance.denominator
                           >= common * concordance.numerator)) | unresolved)
            row, column = torch.where(qualifying)
            pairs = torch.stack((row + start, column,
                                 unresolved[row, column].to(torch.int64)), dim=1).cpu().tolist()
            for left, right, uncertain in pairs:
                if right <= left:
                    continue
                edges += 1
                conservative_edges += uncertain
                first, second = root(left), root(right)
                if first != second:
                    parent[max(first, second)] = min(first, second)
    components = {}
    for index, identity in enumerate(sample_ids):
        components.setdefault(root(index), []).append(identity)
    identifiers = {}
    for members in components.values():
        group_id = hashlib.sha256('\n'.join(sorted(members)).encode()).hexdigest()
        identifiers.update((member, group_id) for member in members)
    return [identifiers[identity] for identity in sample_ids], {
        'samples': n, 'catalog_variants': variants, 'pair_edges': edges,
        'low_coverage_conservative_edges': conservative_edges,
        'samples_below_shared_call_threshold': int(np.sum(
            counts * shared.denominator < variants * shared.numerator)),
        'groups': len(components), 'largest_group': max(map(len, components.values())),
        'min_sample_call_fraction': float(counts.min() / variants),
        'phenotypes_accessed': False, 'global_qc_or_imputation': False,
        'device': str(target), 'accumulation_dtype': 'float32',
    }


def _hash(seed, namespace, identity):
    return hashlib.sha256(f'{seed}\t{namespace}\t{identity}'.encode()).digest()


def _folds(groups, sizes, folds, seed, namespace):
    if folds < 2 or len(groups) < folds:
        raise ValueError('INSUFFICIENT_GROUPS_FOR_FOLDS')
    loads = [0] * folds
    assignment = {}
    for group in sorted(groups, key=lambda g: (-sizes[g], _hash(seed, namespace, g), g)):
        fold = min(range(folds), key=lambda f: (loads[f], f))
        assignment[group] = fold
        loads[fold] += sizes[group]
    return assignment


def plan_partitions(sample_ids, group_ids, sealed_fraction, outer_folds, inner_folds,
                    budgets, seed):
    """Whole-group sealed hash prefix, balanced outer folds and nested budgets.

    Budget selection stops at the first group that would exceed each cap. Inner
    folds are frozen independently for each outer-training budget, using groups
    only. No phenotype is accepted by this function.
    """
    if (len(sample_ids) != len(group_ids) or len(set(sample_ids)) != len(sample_ids)
            or not sample_ids or not 0 < sealed_fraction < 1):
        raise ValueError('PARTITION_COHORT_INVALID')
    if budgets != sorted(set(budgets)) or not budgets or budgets[0] <= 0:
        raise ValueError('TRAINING_BUDGETS_INVALID')
    sizes = Counter(group_ids)
    if outer_folds < 2 or len(sizes) < outer_folds + 1:
        raise ValueError('INSUFFICIENT_GROUPS_FOR_SEALED_AND_OUTER_FOLDS')
    ordered = sorted(sizes, key=lambda g: (_hash(seed, 'sealed', g), g))
    cumulative = np.cumsum([sizes[g] for g in ordered])
    cut = 1 + min(range(len(ordered)-1),
                  key=lambda i: (abs(cumulative[i] - sealed_fraction*len(sample_ids)), i))
    sealed = set(ordered[:cut])
    development = set(ordered[cut:])
    outer = _folds(development, sizes, outer_folds, seed, 'outer')
    partitions = [dict(accession_id=identity, group_id=group,
                       partition='sealed' if group in sealed else 'development',
                       outer_fold='' if group in sealed else outer[group])
                  for identity, group in zip(sample_ids, group_ids)]
    nested, budget_audit = [], []
    for fold in range(outer_folds):
        train = sorted((g for g in development if outer[g] != fold),
                       key=lambda g: (_hash(seed, f'budget:{fold}', g), g))
        previous = set()
        for budget in budgets:
            selected, count = [], 0
            for group in train:
                if count + sizes[group] > budget:
                    break
                selected.append(group)
                count += sizes[group]
            if not previous <= set(selected):
                raise ValueError('TRAINING_BUDGET_NOT_NESTED')
            previous = set(selected)
            inner = _folds(selected, sizes, inner_folds, seed, f'inner:{fold}:{budget}')
            nested.extend(dict(outer_fold=fold, budget=budget, accession_id=identity,
                               group_id=group, inner_fold=inner[group])
                          for identity, group in zip(sample_ids, group_ids) if group in inner)
            budget_audit.append(dict(outer_fold=fold, budget=budget, samples=count,
                                     groups=len(selected), inner_folds=inner_folds))
    return partitions, nested, budget_audit
