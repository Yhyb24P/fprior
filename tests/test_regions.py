import pytest
from preprocessing.regions import exact_hits, mapped_candidates, select_panel


def test_exact_hit_and_secondary_ambiguity(tmp_path):
    path = tmp_path / 'alignments.sam'
    path.write_text('r\t0\tc\t1\t60\t6M\t*\t0\t0\tAAAAAA\t*\tNM:i:0\n'
                    'r\t256\td\t1\t0\t6M\t*\t0\t0\tAAAAAA\t*\tNM:i:0\n'
                    'gap\t0\tc\t1\t60\t3M1I2M\t*\t0\t0\tAAAAAA\t*\tNM:i:1\n')
    hits, queries = exact_hits(path, 6)
    assert len(hits['r']) == 2
    assert not hits['gap']
    assert queries == {'r', 'gap'}


def test_mapping_and_snp_half_open_boundaries(tmp_path):
    source, target = tmp_path / 'source.sam', tmp_path / 'target.sam'
    source.write_text('r\t0\tgnm4.Gm01\t1\t60\t6M\t*\t0\t0\tAAAAAA\t*\tNM:i:0\n')
    target.write_text('r\t0\tgnm2.Gm01\t2\t60\t6M\t*\t0\t0\tAAAAAA\t*\tNM:i:0\n')
    variants = tmp_path / 'variants.tsv'
    variants.write_text('CHROM\tPOS\n' + ''.join(f'gnm2.Gm01\t{x}\n' for x in [1,2,7,8]))
    rows = [{'region_id':'r', 'source_chrom':'gnm4.Gm01', 'source_start':'0', 'source_end':'6'}]
    mapped, rejected = mapped_candidates(rows, source, target, {'gnm4.Gm01':'AAAAAA'},
                                        {'gnm2.Gm01':'AAAAAAA'}, variants, 6, 60)
    assert not rejected
    assert mapped[0]['catalog_snps'] == 2  # POS=2 and POS=7; BED [1,7)
    assert mapped[0]['start'] == 1
    with pytest.raises(ValueError, match='FASTA_SEQUENCE_MISMATCH'):
        mapped_candidates(rows, source, target, {'gnm4.Gm01':'AAAAAA'},
                          {'gnm2.Gm01':'TTTTTTT'}, variants, 6, 60)


def test_panel_order_and_overlap():
    rows = [dict(region_id=str(i), CHROM='c', start=i*3, end=i*3+6, catalog_snps=2)
            for i in range(10)]
    chosen = select_panel(rows, 2, 20, 1)
    assert chosen == select_panel(list(reversed(rows)), 2, 20, 1)
    ordered = sorted(chosen, key=lambda row:row['start'])
    assert all(a['end'] <= b['start'] for a,b in zip(ordered, ordered[1:]))
    assert select_panel([dict(rows[0], catalog_snps=1)], 2, 20, 1) == []
def test_train_window_validity_excludes_missing_and_multiple_unphased_hets():
    import numpy as np
    from preprocessing.regions import audit_window_calls
    x = np.array([[0, 2], [1, 0], [1, 1], [np.nan, 2], [2, 2]])
    records = audit_window_calls(x, {'w': [0, 1]}, [0, 1, 2, 3], .95, .98)
    r = records[0]
    assert r['valid_samples'] == 2 and r['valid_fraction'] == .5
    assert r['missing_samples'] == 1 and r['unresolved_multihet_samples'] == 1
    assert r['qc_snps'] == 1 and not r['passes_valid_fraction']
    # Test rows have no effect on training-derived diagnostics.
    x[4] = [np.nan, np.nan]
    assert audit_window_calls(x, {'w': [0, 1]}, [0, 1, 2, 3], .95, .98) == records
