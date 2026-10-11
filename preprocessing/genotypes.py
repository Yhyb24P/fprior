"""Lossless diploid GT export, without cohort QC, imputation or phenotype access."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re

import numpy as np


def decode_gt(gt: str, allele_count: int) -> tuple[int, int, bool]:
    """Keep partial calls and separator; reject malformed/out-of-range alleles."""
    match = re.fullmatch(r'(\.|\d+)([/|])(\.|\d+)', gt)
    if match is None:
        raise ValueError(f'INVALID_DIPLOID_GT: {gt!r}')
    left, separator, right = match.groups()
    alleles = tuple(-1 if value == '.' else int(value) for value in (left, right))
    if any(value >= allele_count for value in alleles):
        raise ValueError(f'GT_ALLELE_OUT_OF_RANGE: {gt!r}')
    return alleles[0], alleles[1], separator == '|'


def nuclear_snv(chrom: str, ref: str, alt: str) -> bool:
    return bool(re.fullmatch(r'glyma\.Wm82\.gnm2\.Gm(?:0[1-9]|1\d|20)', chrom)) and (
        len(ref) == len(alt) == 1 and ref in 'ACGT' and alt in 'ACGT' and ref != alt)


def export(vcf: Path, destination: Path) -> dict:
    """Write variant-major arrays; raw VCF and all-record GT audit remain available.

    Four arrays share (variant, sample) axes. Alleles retain -1 missing calls,
    phased retains the separator, called requires both alleles. Dosage is NaN
    whenever either allele is missing. No inference of haplotypes is performed.
    """
    vcf, destination = Path(vcf), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = ('dosages.npy', 'alleles.npy', 'phased.npy', 'called.npy',
             'samples.tsv', 'variants.tsv', 'gt_audit.json')
    if any((destination / name).exists() for name in names):
        raise FileExistsError('EXPORT_EXISTS: use a new versioned destination')
    samples = None
    variants = []
    raw_count = 0
    ids = set()
    with gzip.open(vcf, 'rt') as stream:
        for line in stream:
            if line.startswith('#CHROM\t'):
                samples = line.rstrip('\n').split('\t')[9:]
            elif not line.startswith('#'):
                fields = line.split('\t', 5)
                chrom, pos, identity, ref, alt = fields[:5]
                raw_count += 1
                if identity == '.' or identity in ids:
                    raise ValueError('VARIANT_ID_MISSING_OR_DUPLICATE')
                ids.add(identity)
                if nuclear_snv(chrom, ref, alt):
                    variants.append((chrom, int(pos), identity, ref, alt))
    if not samples or len(set(samples)) != len(samples) or any(not s for s in samples):
        raise ValueError('SAMPLE_ID_INVALID_OR_DUPLICATE')
    if not variants:
        raise ValueError('NO_NUCLEAR_BIALLELIC_SNVS')
    shape = (len(variants), len(samples))
    arrays = {
        'dosages': np.lib.format.open_memmap(destination / 'dosages.npy', mode='w+', dtype='float32', shape=shape),
        'alleles': np.lib.format.open_memmap(destination / 'alleles.npy', mode='w+', dtype='int8', shape=shape + (2,)),
        'phased': np.lib.format.open_memmap(destination / 'phased.npy', mode='w+', dtype='bool', shape=shape),
        'called': np.lib.format.open_memmap(destination / 'called.npy', mode='w+', dtype='bool', shape=shape),
    }
    audit = Counter()
    kept_audit = Counter()
    index = 0
    try:
        with gzip.open(vcf, 'rt') as stream:
            for line in stream:
                if line.startswith('#'):
                    continue
                fields = line.rstrip('\n').split('\t')
                if len(fields) != 9 + len(samples):
                    raise ValueError('VCF_SAMPLE_COLUMN_MISMATCH')
                chrom, pos, identity, ref, alt = fields[:5]
                formats = fields[8].split(':')
                if formats.count('GT') != 1:
                    raise ValueError('VCF_GT_FORMAT_REQUIRED')
                gt_index = formats.index('GT')
                calls = [sample.split(':')[gt_index] for sample in fields[9:]]
                counts = Counter(calls)
                decoded = {gt: decode_gt(gt, len(alt.split(',')) + 1) for gt in counts}
                audit.update(counts)
                if not nuclear_snv(chrom, ref, alt):
                    continue
                if variants[index] != (chrom, int(pos), identity, ref, alt):
                    raise ValueError('VCF_CHANGED_DURING_EXPORT')
                kept_audit.update(counts)
                values = np.asarray([decoded[gt] for gt in calls], dtype=np.int8)
                called = (values[:, :2] >= 0).all(axis=1)
                dosage = values[:, :2].sum(axis=1).astype(np.float32)
                dosage[~called] = np.nan
                arrays['dosages'][index] = dosage
                arrays['alleles'][index] = values[:, :2]
                arrays['phased'][index] = values[:, 2]
                arrays['called'][index] = called
                index += 1
        if index != len(variants):
            raise ValueError('VCF_CHANGED_DURING_EXPORT')
        for array in arrays.values():
            array.flush()
        with (destination / 'samples.tsv').open('w') as stream:
            writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
            writer.writerow(('sample_index', 'raw_id', 'normalized_id', 'evidence'))
            writer.writerows((i, s, s, 'VCF header; identity normalization') for i, s in enumerate(samples))
        with (destination / 'variants.tsv').open('w') as stream:
            writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
            writer.writerow(('CHROM', 'POS', 'ID', 'REF', 'ALT'))
            writer.writerows(variants)
        result = {'source': str(vcf), 'axes': ['variant', 'sample'],
                  'samples': len(samples), 'raw_variants': raw_count,
                  'retained_variants': index, 'excluded_variants': raw_count-index,
                  'raw_gt_counts': dict(audit), 'retained_gt_counts': dict(kept_audit),
                  'dosage': 'ALT count; partial or full missing => NaN',
                  'phase': 'separator retained; phased AND called required for phase validity',
                  'phenotypes_accessed': False, 'qc_or_imputation_applied': False}
        (destination / 'gt_audit.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    except Exception:
        # Only files created by this failed invocation; never leave usable partial arrays.
        arrays.clear()
        for name in names:
            (destination / name).unlink(missing_ok=True)
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    from fprior.io import load_config
    config, root = load_config(args.config)
    print(json.dumps(export(root / config['data']['raw_calls'], root / config['data']['directory']), indent=2))
