"""Exact reference-anchored D/A/U/R algebra and trustworthy window haplotypes."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .io import finite_array


@dataclass(frozen=True)
class WindowVariant:
    index: int
    offset: int  # zero-based offset within the window
    ref: str
    alt: str


def _variants(reference: str, variants: list[WindowVariant], n_variants: int):
    if not reference or set(reference) - set('ACGT'):
        raise ValueError('REFERENCE_SEQUENCE_INVALID')
    indices, offsets = set(), set()
    for variant in variants:
        if (not isinstance(variant.index, (int, np.integer))
                or not isinstance(variant.offset, (int, np.integer))
                or variant.index in indices or variant.offset in offsets
                or not 0 <= variant.index < n_variants
                or not 0 <= variant.offset < len(reference)):
            raise ValueError('VARIANT_MAPPING_INVALID')
        if variant.ref not in 'ACGT' or variant.alt not in 'ACGT' or len(variant.ref) != 1 or len(variant.alt) != 1 or variant.ref == variant.alt:
            raise ValueError('VARIANT_NOT_BIALLELIC_SNV')
        if reference[variant.offset] != variant.ref:
            raise ValueError('REFERENCE_ALLELE_MISMATCH')
        indices.add(variant.index)
        offsets.add(variant.offset)


def resolve_haplotypes(dosages, reference: str, variants: list[WindowVariant], *, phased_alleles=None):
    """Missing or multiple unphased heterozygotes return None, never invented phase."""
    q = np.asarray(dosages, dtype=np.float64)
    if q.ndim != 1 or not np.isin(q[~np.isnan(q)], [0, 1, 2]).all():
        raise ValueError('DOSAGE_INVALID')
    _variants(reference, variants, len(q))
    indices = [v.index for v in variants]
    window = q[indices]
    if np.isnan(window).any():
        return None
    if phased_alleles is None:
        if np.count_nonzero(window == 1) > 1:
            return None
        alleles = np.column_stack((window >= 1, window == 2))
    else:
        phase = np.asarray(phased_alleles, dtype=np.float64)
        if phase.shape != (len(q), 2):
            raise ValueError('PHASE_SHAPE_INVALID')
        alleles = phase[indices]
        if not np.isin(alleles[~np.isnan(alleles)], [0, 1]).all():
            raise ValueError('PHASE_ALLELE_INVALID')
        if np.isnan(alleles).any():
            return None
        if not np.array_equal(alleles.sum(axis=1), window):
            raise ValueError('PHASE_DOSAGE_MISMATCH')
    sequences = [list(reference), list(reference)]
    for j, variant in enumerate(variants):
        for chromosome in range(2):
            if alleles[j, chromosome]:
                sequences[chromosome][variant.offset] = variant.alt
    return tuple(''.join(sequence) for sequence in sequences)


def build_d_matrix(teacher, references: list[str], variants_by_window: list[list[WindowVariant]], *, n_variants: int) -> np.ndarray:
    if n_variants < 1 or len(references) != len(variants_by_window):
        raise ValueError('WINDOW_MAPPING_INVALID')
    d = np.zeros((n_variants, len(references)), dtype=np.float64)
    for g, (reference, variants) in enumerate(zip(references, variants_by_window)):
        _variants(reference, variants, n_variants)
        sequences = [reference]
        for variant in variants:
            sequence = list(reference)
            sequence[variant.offset] = variant.alt
            sequences.append(''.join(sequence))
        outputs = finite_array(teacher.score(sequences), 1, 'teacher outputs')
        if len(outputs) != len(sequences):
            raise ValueError('TEACHER_OUTPUT_SHAPE_INVALID')
        for j, variant in enumerate(variants):
            d[variant.index, g] = outputs[j + 1] - outputs[0]
    return d


def boolean_coefficients(outputs):
    """Exact Möbius coefficients of F on the complete Boolean haplotype cube."""
    values = finite_array(outputs, 1, 'haplotype teacher outputs').copy()
    size = len(values)
    if size < 4 or size & (size-1):
        raise ValueError('HAPLOTYPE_CUBE_INCOMPLETE')
    for bit in range(size.bit_length()-1):
        for mask in range(size):
            if mask & (1 << bit):
                values[mask] -= values[mask ^ (1 << bit)]
    return values


def local_interaction_counts(dosages, windows, *, phased_alleles=None):
    """All order >=2 haplotype co-occurrences; invalid windows are raw zero.

    The mask and raw-zero rule match R exactly, including coverage structure.
    No categories, weights or means are learned from test rows. Subset ordering
    follows increasing binary mask, with the first SNP as the highest bit.
    """
    q = np.asarray(dosages, dtype=np.float64)
    if q.ndim != 2 or not np.isin(q[~np.isnan(q)], [0, 1, 2]).all():
        raise ValueError('DOSAGE_INVALID')
    if phased_alleles is not None:
        phase = np.asarray(phased_alleles, dtype=np.float64)
        if phase.shape != (*q.shape, 2) or not np.isin(phase[~np.isnan(phase)], [0, 1]).all():
            raise ValueError('PHASE_SHAPE_OR_ALLELE_INVALID')
    blocks, masks, subsets = [], [], []
    for columns in windows:
        if len(columns) < 2 or len(columns) > 8 or len(set(columns)) != len(columns):
            raise ValueError('INTERACTION_WINDOW_INVALID')
        calls = q[:, columns]
        valid = np.isfinite(calls).all(axis=1)
        if phased_alleles is None:
            valid &= (calls == 1).sum(axis=1) <= 1
            alleles = np.stack([calls >= 1, calls == 2], axis=-1)
        else:
            alleles = phase[:, columns]
            phase_called = np.isfinite(alleles).all(axis=(1, 2))
            if not np.array_equal(alleles[valid & phase_called].sum(axis=2), calls[valid & phase_called]):
                raise ValueError('PHASE_DOSAGE_MISMATCH')
            valid &= phase_called
        selected = [m for m in range(1 << len(columns)) if m.bit_count() >= 2]
        block = np.zeros((len(q), len(selected)))
        for j, subset in enumerate(selected):
            indices = [i for i in range(len(columns)) if subset & (1 << (len(columns)-1-i))]
            block[valid, j] = np.all(alleles[valid][:, indices] == 1, axis=1).sum(axis=1)
        blocks.append(block); masks.append(valid); subsets.append(selected)
    if not blocks:
        raise ValueError('INTERACTION_WINDOWS_MISSING')
    return blocks, np.column_stack(masks), subsets


def permuted_residual_features(blocks, coefficients, rng):
    """Permute full teacher coefficient vectors among equal-arity windows only."""
    if len(blocks) != len(coefficients) or not blocks:
        raise ValueError('INTERACTION_COEFFICIENT_ALIGNMENT_INVALID')
    strata = {}
    for g, (block, coefficient) in enumerate(zip(blocks, coefficients)):
        if block.ndim != 2 or len(coefficient) != block.shape[1]:
            raise ValueError('INTERACTION_COEFFICIENT_SHAPE_INVALID')
        strata.setdefault(block.shape[1], []).append(g)
    assignment = np.arange(len(blocks))
    for indices in strata.values():
        assignment[indices] = rng.permutation(indices)
    values = np.column_stack([block@coefficients[assignment[g]] for g, block in enumerate(blocks)])
    return values, assignment


def build_features(dosages, d_matrix, functional_outputs, valid_mask):
    """functional_outputs is raw U, with the diploid reference already subtracted."""
    q = finite_array(dosages, 2, 'imputed dosages')
    d = finite_array(d_matrix, 2, 'D')
    if q.shape[1] != d.shape[0] or ((q < 0) | (q > 2)).any():
        raise ValueError('DOSAGE_D_MAPPING_INVALID')
    a = q @ d
    u = np.asarray(functional_outputs, dtype=np.float64)
    mask = np.asarray(valid_mask)
    if u.shape != a.shape or mask.shape != a.shape or mask.dtype != np.bool_:
        raise ValueError('FUNCTIONAL_SHAPE_INVALID')
    if not np.isfinite(u[mask]).all():
        raise ValueError('TEACHER_OUTPUT_NONFINITE')
    u = np.where(mask, u, a)
    r = u - a
    return a, u, r, mask.copy()


@dataclass(frozen=True)
class ColumnCenterer:
    mean: np.ndarray

    @classmethod
    def fit(cls, train):
        train = finite_array(train, 2, 'training features')
        if not len(train):
            raise ValueError('EMPTY_TRAINING_PARTITION')
        return cls(train.mean(axis=0))

    def transform(self, values):
        values = finite_array(values, 2, 'features')
        if values.shape[1] != len(self.mean):
            raise ValueError('FEATURE_ORDER_MISMATCH')
        return values - self.mean


def center_trainwise(features, train_idx):
    features = finite_array(features, 2, 'features')
    indices = np.asarray(train_idx)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices) or len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= len(features)).any():
        raise ValueError('TRAINING_INDEX_INVALID')
    return ColumnCenterer.fit(features[indices]).transform(features)


def verify_identities(x, d, z, u, r, *, relative_tolerance: float) -> float:
    arrays = [finite_array(value, 2, name) for value, name in zip((x, d, z, u, r), ('X', 'D', 'Z', 'U', 'R'))]
    x, d, z, u, r = arrays
    if x.shape[1] != d.shape[0] or z.shape != (len(x), d.shape[1]) or u.shape != z.shape or r.shape != z.shape:
        raise ValueError('IDENTITY_SHAPE_INVALID')
    error = max(np.max(np.abs(z - x @ d), initial=0), np.max(np.abs(u - z - r), initial=0))
    scale = max(np.max(np.abs(value), initial=0) for value in (z, u, r))
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0 or error > relative_tolerance * (1 + scale):
        raise ValueError(f'FEATURE_IDENTITY_FAILED: max_abs_error={error}')
    return float(error)
