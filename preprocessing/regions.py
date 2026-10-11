"""Exact reference-window transfer and phenotype-free SNP coverage selection."""
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path


def exact_hits(path: Path, length: int):
    """Retain full, gap-free, zero-edit SAM hits; keep secondary hits for ambiguity."""
    hits = defaultdict(list)
    queries = set()
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith('@'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 11:
                raise ValueError('SAM_RECORD_INVALID')
            identity, flag = fields[0], int(fields[1])
            queries.add(identity)
            if flag & (4 | 2048):
                continue
            if fields[5] != f'{length}M' or 'NM:i:0' not in fields[11:]:
                continue
            hits[identity].append((fields[2], int(fields[3])-1, int(fields[4]),
                                   bool(flag & 16), bool(flag & 256)))
    return hits, queries


def mapped_candidates(source_rows, source_sam, target_sam, source_fasta,
                      target_fasta, variant_table, length: int, min_mapq: int):
    """Use aligner-reported unique exact hits plus independent FASTA identity.

    Transfer the entire predeclared source-centered window. The resulting BED
    represents that window, with original peak coordinates retained as provenance.
    No remidpointing, sequence repair, frequency filtering or phenotype access.
    """
    sources = {row['region_id']: row for row in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError('REGION_ID_DUPLICATE')
    source_hits, seen_source = exact_hits(source_sam, length)
    target_hits, seen_target = exact_hits(target_sam, length)
    if seen_source != set(sources) or seen_target != set(sources):
        raise ValueError('ALIGNMENT_QUERY_SET_MISMATCH')
    snps = defaultdict(list)
    with Path(variant_table).open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            snps[row['CHROM']].append(int(row['POS'])-1)
    for positions in snps.values():
        positions.sort()
    rejected = Counter()
    retained = []
    for identity, row in sources.items():
        source, target = source_hits[identity], target_hits[identity]
        source_start, source_end = int(row['source_start']), int(row['source_end'])
        if source_end-source_start != length:
            raise ValueError('SOURCE_WINDOW_LENGTH_MISMATCH')
        if (len(source) != 1 or source[0][2] < min_mapq or source[0][3]
                or source[0][4] or source[0][:2] != (row['source_chrom'], source_start)):
            rejected['source_not_unique_exact_original_locus'] += 1
            continue
        if len(target) != 1 or target[0][2] < min_mapq or target[0][4]:
            rejected['target_not_unique_exact_full_window'] += 1
            continue
        chrom, start, mapq, reverse, _ = target[0]
        end = start+length
        if chrom.split('.')[-1] != row['source_chrom'].split('.')[-1]:
            rejected['target_different_chromosome'] += 1
            continue
        sequence = target_fasta[chrom][start:end]
        if reverse:
            sequence = sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
        original = source_fasta[row['source_chrom']][source_start:source_end]
        if len(sequence) != length or sequence != original:
            raise ValueError('MAPPING_FASTA_SEQUENCE_MISMATCH')
        positions = snps[chrom]
        count = bisect_left(positions, end)-bisect_left(positions, start)
        retained.append(dict(row, CHROM=chrom, start=start, end=end,
                             strand='-' if reverse else '+', mapq=mapq,
                             catalog_snps=count))
    return retained, dict(rejected)


def select_panel(candidates, min_snps: int, max_windows: int, seed: int):
    """Hash the source region identity, then greedily retain nonoverlapping windows."""
    if min_snps < 2 or max_windows <= 0:
        raise ValueError('REGION_SELECTION_POLICY_INVALID')
    ordered = sorted((row for row in candidates if row['catalog_snps'] >= min_snps),
                     key=lambda row: (hashlib.sha256(f"{seed}\t{row['region_id']}".encode()).digest(),
                                      row['region_id']))
    intervals = defaultdict(list)
    selected = []
    for row in ordered:
        existing = intervals[row['CHROM']]
        place = bisect_left(existing, (row['start'], row['end']))
        if place and existing[place-1][1] > row['start']:
            continue
        if place < len(existing) and existing[place][0] < row['end']:
            continue
        existing.insert(place, (row['start'], row['end']))
        selected.append(row)
        if len(selected) == max_windows:
            break
    return selected
