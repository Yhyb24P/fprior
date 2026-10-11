from pathlib import Path
import pytest
from preprocessing.phenotypes import protein_study, method_document, aggregate_studies


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


def test_equal_study_weights_and_ambiguity():
    def row(study, value, numeric=True):
        return dict(accession_id='PI1', grin_accession_id='1', study_id=study,
                    raw_value=str(value), finite_numeric_value=numeric)
    records = [row('a', 40), row('a', 44), row('b', 48), row('excluded', 0),
               row('missing', 'NA', False)]
    result, = aggregate_studies(records, ['excluded'])
    assert result['protein'] == 45  # mean(mean(40,44),48); not mean(40,44,48)
    assert result['study_count'] == 2
    assert result['published_value_count'] == 3
    with pytest.raises(ValueError, match='ACCESSION_AMBIGUITY'):
        aggregate_studies(records+[dict(records[0], grin_accession_id='2')], [])


def test_method_document_name_is_authoritative(tmp_path):
    path = tmp_path/'method.html'
    path.write_text('<span id="MainContent_lblName">Study A</span>'
                    '<table id="MainContent_dvMethod"><tr><td>Seeds measured using NIR.</td></tr></table>')
    assert method_document(path, '1', 'Study A')['published_description'].endswith('NIR.')
    with pytest.raises(ValueError, match='METHOD_NAME_MISMATCH'):
        method_document(path, '1', 'Study B')
