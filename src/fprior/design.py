"""Fit a common, training-only design for the registered model comparisons."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from preprocessing.dosages import DosageTransform
from .features import (boolean_coefficients, local_interaction_counts,
                       permuted_residual_features, ColumnCenterer, verify_identities)
from .kernels import (trace_normalize, cross_kernel, GaussianKernel,
                      haplotype_kernel, local_rbf_kernels)


@dataclass
class FoldDesign:
    kernels: dict
    families: dict
    features: dict
    diagnostics: dict


def build_design(train, test, windows, outputs, config, *, noise_threshold):
    """No labels accepted; test rows cannot influence any fitted statistic.

    outputs contains complete Boolean teacher cubes in physical SNP bit order.
    This implementation deliberately rejects phased sources at its input reader;
    it resolves only fully called windows with at most one heterozygous SNP.
    """
    policy = config['variant_policy']
    transform = DosageTransform.fit(train, min_call_rate=policy['min_train_call_rate'],
                                   maf_min=policy['maf_min'])
    qt, xt = transform.transform(train)
    qe, xe = transform.transform(test)
    blocks_t, valid_t, subsets = local_interaction_counts(train, windows)
    blocks_e, valid_e, _ = local_interaction_counts(test, windows)
    usable = [g for g, columns in enumerate(windows)
              if transform.retained[columns].all()
              and valid_t[:, g].mean() >= config['regions']['min_train_valid_fraction']]
    if len(usable) < config['regions']['min_usable_windows']:
        raise ValueError('INSUFFICIENT_USABLE_WINDOWS')
    retained_indices = np.flatnonzero(transform.retained)
    index = {old: new for new, old in enumerate(retained_indices)}
    d = np.zeros((xt.shape[1], len(usable)))
    coefficients = []
    for j, g in enumerate(usable):
        coef = boolean_coefficients(outputs[g])
        if len(coef) != 1 << len(windows[g]):
            raise ValueError('TEACHER_CUBE_WINDOW_MISMATCH')
        for i, column in enumerate(windows[g]):
            d[index[column], j] = coef[1 << (len(windows[g])-1-i)]
        coefficients.append(coef[subsets[g]])
    bt = [blocks_t[g] for g in usable]; be = [blocks_e[g] for g in usable]
    at, ae = qt@d, qe@d
    rt = np.column_stack([b@c for b, c in zip(bt, coefficients)])
    re = np.column_stack([b@c for b, c in zip(be, coefficients)])
    raw = {'F': (at, ae), 'U': (at+rt, ae+re), 'R': (rt, re),
           'I': (np.concatenate(bt, axis=1), np.concatenate(be, axis=1))}
    features = {'X': (xt, xe)}
    for name, (a, b) in raw.items():
        center = ColumnCenterer.fit(a)
        features[name] = (center.transform(a), center.transform(b))
    error = verify_identities(xt, d, features['F'][0], features['U'][0],
                              features['R'][0], relative_tolerance=
                              config['solver']['feature_identity_relative_tolerance'])
    kernels = {}; diagnostics = {'usable_windows': usable,
        'retained_snps': len(retained_indices), 'feature_identity_error': error,
        'valid_fraction': valid_t[:, usable].mean(axis=0).tolist(),
        'coefficient_vectors': [c.tolist() for c in coefficients],
        'D_singleton_effects': [d[[index[i] for i in windows[g]], j].tolist()
                               for j, g in enumerate(usable)],
        'blocks': {}, 'MP_changed_fraction': [], 'MRPerm_assignments': []}

    def linear(name, pair, *, functional=False, gram=None, cross_gram=None):
        a, b = pair
        rms = float(np.sqrt(np.mean(a*a))) if a.size else 0.
        if gram is None:
            k, scale = trace_normalize(a)
            cross = cross_kernel(b, a, scale)
        else:
            scale = float(np.sum(a*a)/len(a))
            k = gram/scale if scale else np.zeros_like(gram)
            cross = cross_gram/scale if scale else np.zeros_like(cross_gram)
        if functional and rms <= noise_threshold:
            k[:] = 0; cross[:] = 0
        denominator = float(np.sum(k*k))
        diagnostics['blocks'][name] = {'rms': rms, 'trace_scale': scale,
            'column_rms': np.sqrt(np.mean(a*a, axis=0)).tolist(),
            'signal_resolvable': bool(rms > noise_threshold),
            'participation_rank': float(np.trace(k)**2/denominator) if denominator else 0.,
            'near_zero_columns': int(np.sum(np.sqrt(np.mean(a*a, axis=0)) <= noise_threshold))}
        kernels[name] = (k, cross)

    if config['execution'].get('compute_device', 'cpu') == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA_UNAVAILABLE')
        gt = torch.as_tensor(xt, dtype=torch.float64, device='cuda')
        ge = torch.as_tensor(xe, dtype=torch.float64, device='cuda')
        raw_gram = (gt@gt.T).cpu().numpy()
        raw_cross = (ge@gt.T).cpu().numpy()
        del gt, ge
    else:
        raw_gram = xt@xt.T; raw_cross = xe@xt.T
    diagnostics['compute_device'] = config['execution'].get('compute_device', 'cpu')
    linear('A', features['X'], gram=raw_gram, cross_gram=raw_cross)
    for name in ('F', 'U', 'R', 'I'):
        linear(name, features[name], functional=name != 'I')
    diagnostics['functional_status'] = ('SIGNAL_RESOLVABLE' if kernels['F'][0].any() or kernels['R'][0].any()
                                          else 'NO_RESOLVABLE_SIGNAL')
    annotation = np.sqrt(np.sum(d*d, axis=1)); annotated = annotation > 0
    linear('D', (xt[:, annotated]*annotation[annotated], xe[:, annotated]*annotation[annotated]), functional=True)
    for name, pair, gram, cross in [('GX', features['X'], raw_gram, raw_cross),
                                   ('GU', features['U'], None, None)]:
        fitted = GaussianKernel.fit(pair[0], raw_gram=gram)
        kernels[name] = (fitted.normalized_train, fitted.transform(pair[1], raw_cross=cross))
        diagnostics[name+'_bandwidth'] = fitted.bandwidth
    local_columns = [[index[i] for i in windows[g]] for g in usable]
    kl, kle, removed = local_rbf_kernels([xt[:, c] for c in local_columns],
                                        [xe[:, c] for c in local_columns])
    # Explicit final normalization also removes accumulation roundoff.
    local_scale = float(np.trace(kl)/len(kl))
    if local_scale > 0:
        kl /= local_scale; kle /= local_scale
    kernels['Local'] = (kl, kle); diagnostics['local_zero_windows'] = removed

    def hap_counts(q, mask, columns):
        calls = q[:, columns]
        h = np.stack([calls >= 1, calls == 2], axis=-1)
        codes = np.sum(h*np.asarray([1 << (len(columns)-1-i)
                       for i in range(len(columns))])[None, :, None], axis=1)
        counts = np.column_stack([(codes == value).sum(axis=1)
                                  for value in range(1 << len(columns))]).astype(float)
        counts[~mask] = np.nan
        return counts
    ht = [hap_counts(train, valid_t[:, g], windows[g]) for g in usable]
    he = [hap_counts(test, valid_e[:, g], windows[g]) for g in usable]
    kernels['Hap'] = haplotype_kernel(np.concatenate(ht, axis=1), np.concatenate(he, axis=1))

    families = {'M0': ['A'], 'MD': ['A', 'D'], 'M1': ['A', 'F'],
        'M2': ['A', 'F', 'R'], 'MU': ['A', 'U'], 'MW': ['A', 'GU'],
        'MWG': ['GX', 'GU'], 'MHap': ['A', 'Hap'], 'MLocal': ['A', 'Local'],
        'FU': ['U'], 'MI': ['A', 'I'], 'MI3': ['A', 'F', 'I']}
    controls = config['controls']
    edges = np.asarray(controls['maf_bin_edges'])
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or edges[-1] != .5 or (np.diff(edges) <= 0).any():
        raise ValueError('MAF_BINS_INVALID')
    for repeat in range(controls['permutation_replicates']):
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], 0, repeat]))
        dp = d.copy(); changed = 0; total = 0
        for j, g in enumerate(usable):
            original = np.asarray(windows[g]); columns = np.asarray(local_columns[j])
            bins = np.minimum(np.searchsorted(edges, transform.maf[original], side='right')-1,
                              len(edges)-2)
            for label in np.unique(bins):
                select = columns[bins == label]
                permutation = rng.permutation(select)
                dp[select, j] = d[permutation, j]
                changed += np.sum(dp[select, j] != d[select, j]); total += len(select)
        name = f'FP:{repeat}'
        linear(name, (xt@dp, xe@dp), functional=True)
        families[f'MP:{repeat}'] = ['A', name]
        diagnostics['MP_changed_fraction'].append(float(changed/total) if total else 0.)
    for repeat in range(controls['residual_weight_permutation_replicates']):
        seed = np.random.SeedSequence([config['seed'], controls['residual_weight_permutation_control_id'], repeat])
        pt, assignment = permuted_residual_features(bt, coefficients, np.random.default_rng(seed))
        pe = np.column_stack([be[g]@coefficients[assignment[g]] for g in range(len(be))])
        center = ColumnCenterer.fit(pt); name = f'RP:{repeat}'
        linear(name, (center.transform(pt), center.transform(pe)), functional=True)
        families[f'MRPerm:{repeat}'] = ['A', name]
        families[f'MRPerm3:{repeat}'] = ['A', 'F', name]
        diagnostics['MRPerm_assignments'].append(assignment.tolist())
    names = list(kernels)
    vectors = np.stack([kernels[name][0].ravel() for name in names])
    norms = np.linalg.norm(vectors, axis=1)
    denominator = norms[:, None]*norms
    alignment = np.divide(vectors@vectors.T, denominator,
                          out=np.zeros_like(denominator), where=denominator > 0)
    diagnostics['kernel_alignment'] = {'names': names, 'centered_frobenius_cosine': alignment.tolist()}
    diagnostics['kernel_spectra'] = {}
    for name in names:
        k = kernels[name][0]
        eigenvalues = np.linalg.eigvalsh((k+k.T)/2)
        diagnostics['kernel_spectra'][name] = {'trace': float(np.trace(k)),
            'minimum_eigenvalue': float(eigenvalues.min()),
            'maximum_eigenvalue': float(eigenvalues.max()),
            'alignment_with_KA': float(alignment[names.index(name), names.index('A')])}
    return FoldDesign(kernels, families, features, diagnostics)
