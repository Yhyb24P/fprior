from pathlib import Path
import pytest
from preprocessing.phenotypes import protein_study


def fixture(path: Path, value='47.2'):
    headers = ['', 'ID', 'ACCESSION', 'NAME', 'TAXONOMY', 'ORIGIN',
               'AVAILABILITY', 'IMAGE', 'VALUE', 'OBSERVATIONS', 'NOTE', 'INVENTORY']
    row = ['0', '1486418', 'PI 407859 -2', '', 'Glycine max', '', '', '',
           value, '0', 'Data from 1998 only', 'PI 407859 -2 **']
    path.write_text('<form action="methodaccession?id1=51016&amp;id2=493088">'
                    "<table id='searchtable'><tr>" + ''.join(f'<th>{x}</th>' for x in headers)
                    + '</tr><tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr></table></form>')


def test_raw_observation_provenance(tmp_path):
    path = tmp_path / 'study.html'
    fixture(path)
    record, = protein_study(path, '493088', 'SOJA.EVALUATION.IL989')
    assert record['accession_id'] == 'PI407859-2'
    assert record['inventory_raw'].endswith('**')
    assert record['observations_metadata'] == '0'
    assert record['raw_value'] == '47.2'
    assert record['finite_numeric_value']
    assert record['aggregation_rule'].startswith('none')
    with pytest.raises(ValueError, match='CONTEXT_MISMATCH'):
        protein_study(path, '494186', 'wrong study')


def test_non_numeric_is_not_imputed(tmp_path):
    path = tmp_path / 'study.html'
    fixture(path, 'not evaluated')
    record, = protein_study(path, '493088', 'SOJA.EVALUATION.IL989')
    assert not record['finite_numeric_value']
    assert record['raw_value'] == 'not evaluated'


def test_grin_extra_closing_cell_preserves_columns(tmp_path):
    path = tmp_path / 'study.html'
    fixture(path)
    path.write_text(path.read_text().replace('<td>47.2</td>', '<td>47.2</td></td>'))
    record, = protein_study(path, '493088', 'SOJA.EVALUATION.IL989')
    assert record['raw_value'] == '47.2'
    assert record['observations_metadata'] == '0'
