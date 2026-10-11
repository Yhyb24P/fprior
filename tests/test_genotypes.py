"""VCF semantics, excluded-call audits and output alignment."""
import gzip
import json
import numpy as np
import pytest
from preprocessing.genotypes import decode_gt, export


@pytest.mark.parametrize('gt,expected', [('0/1',(0,1,False)), ('1/0',(1,0,False)),
    ('1|0',(1,0,True)), ('./.',(-1,-1,False)), ('0|.',(0,-1,True))])
def test_gt(gt, expected):
    assert decode_gt(gt, 2) == expected


@pytest.mark.parametrize('gt', ['0', '0/1/1', '0/2', '-1/0', '0||1'])
def test_invalid_gt(gt):
    with pytest.raises(ValueError):
        decode_gt(gt, 2)


def test_export(tmp_path):
    source = tmp_path / 'source.vcf.gz'
    with gzip.open(source, 'wt') as stream:
        stream.write('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tPI2\tPI1\n')
        stream.write('glyma.Wm82.gnm2.Gm01\t1\tss1\tA\tG\t.\t.\t.\tGT\t1/0\t0|.\n')
        stream.write('glyma.Wm82.gnm2.scaffold_1\t2\tss2\tA\tG,T\t.\t.\t.\tGT\t2/2\t./.\n')
    destination = tmp_path / 'output'
    result = export(source, destination)
    assert result['retained_variants'] == 1
    assert result['raw_gt_counts']['2/2'] == 1
    assert np.load(destination / 'dosages.npy')[0, 0] == 1
    assert np.isnan(np.load(destination / 'dosages.npy')[0, 1])
    assert np.load(destination / 'alleles.npy')[0, 1].tolist() == [0, -1]
    assert np.load(destination / 'phased.npy')[0].tolist() == [False, True]
    assert np.load(destination / 'called.npy')[0].tolist() == [True, False]
    assert (destination / 'samples.tsv').read_text().splitlines()[1].split('\t')[1] == 'PI2'
    assert json.loads((destination / 'gt_audit.json').read_text())['qc_or_imputation_applied'] is False
    with pytest.raises(FileExistsError):
        export(source, destination)


@pytest.mark.parametrize('chrom,gt', [('scaffold', '0/0'),
    ('glyma.Wm82.gnm2.Gm01', '0/2')])
def test_failed_export_has_no_formal_products(tmp_path, chrom, gt):
    source = tmp_path / 'source.vcf.gz'
    with gzip.open(source, 'wt') as stream:
        stream.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tPI1\n')
        stream.write(f'{chrom}\t1\tss1\tA\tG\t.\t.\t.\tGT\t{gt}\n')
    destination = tmp_path / 'output'
    with pytest.raises(ValueError):
        export(source, destination)
    assert list(destination.iterdir()) == []
