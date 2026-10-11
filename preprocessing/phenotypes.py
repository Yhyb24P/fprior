"""Parse official GRIN per-study values, retaining observation provenance."""
from html.parser import HTMLParser
from html import unescape
from collections import defaultdict
import math
from pathlib import Path
import re


class StudyTable(HTMLParser):
    def __init__(self, table_id='searchtable'):
        super().__init__()
        self.table_id = table_id
        self.active = False
        self.cell = None
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == 'table' and dict(attrs).get('id') == self.table_id:
            self.active = True
        if self.active and tag == 'tr':
            self.row = []
        if self.active and tag in ('td', 'th'):
            self.cell = ''

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if self.active and tag in ('td', 'th'):
            # GRIN emits an extra </td> after some unavailable-inventory cells.
            # Like the browser, ignore unmatched closing tags; schema stays exact.
            if self.cell is not None:
                self.row.append(self.cell.strip())
            self.cell = None
        if self.active and tag == 'tr':
            self.rows.append(self.row)
        if self.active and tag == 'table':
            self.active = False


def protein_study(path: Path, study_id: str, study_name: str) -> list[dict]:
    """Published study VALUE is retained; individual replicate means are not inferred.

    Zero OBSERVATIONS is preserved as source metadata, not used to discard VALUE.
    Inventory text is not reinterpreted as a numeric inventory database key.
    """
    html = Path(path).read_text()
    if not re.search(rf'id1=51016(?:&|&amp;)id2={re.escape(study_id)}(?:[\s\"\'<>]|$)', html):
        raise ValueError('PROTEIN_STUDY_CONTEXT_MISMATCH')
    table = StudyTable()
    table.feed(html)
    expected = ['', 'ID', 'ACCESSION', 'NAME', 'TAXONOMY', 'ORIGIN',
                'AVAILABILITY', 'IMAGE', 'VALUE', 'OBSERVATIONS', 'NOTE', 'INVENTORY']
    if not table.rows or table.rows[0] != expected:
        raise ValueError('PROTEIN_STUDY_TABLE_SCHEMA_MISMATCH')
    records = []
    for row in table.rows[1:]:
        if len(row) != len(expected):
            raise ValueError('PROTEIN_STUDY_ROW_SCHEMA_MISMATCH')
        accession = re.sub(r'\s+', '', row[2])
        if not accession or not row[1].isdigit():
            raise ValueError('PROTEIN_ACCESSION_ID_INVALID')
        try:
            value = float(row[8])
        except ValueError:
            value = math.nan
        records.append({'accession_id': accession, 'accession_raw': row[2],
                        'grin_accession_id': row[1], 'descriptor_id': '51016',
                        'study_id': study_id, 'study_name': study_name,
                        'inventory_raw': row[11], 'raw_value': row[8],
                        'finite_numeric_value': math.isfinite(value),
                        'observations_metadata': row[9], 'observation_note': row[10],
                        'unit': '% seed dry weight',
                        'source': f'https://npgsweb.ars-grin.gov/gringlobal/methodaccession.aspx?id1=51016&id2={study_id}',
                        'value_level': 'official study-accession VALUE',
                        'aggregation_rule': 'none; raw study values retained',
                        'transform_rule': 'none', 'imputation_flag': 'not imputed by fprior'})
    identities = [(r['grin_accession_id'], r['inventory_raw']) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError('PROTEIN_STUDY_DUPLICATE_IDENTITY')
    return records


def method_document(path: Path, study_id: str, expected_name: str) -> dict:
    html = Path(path).read_text()
    name = re.search(r'<span id="MainContent_lblName"[^>]*>(.*?)</span>', html, re.S)
    if name is None or unescape(name.group(1)).strip() != expected_name:
        raise ValueError('METHOD_NAME_MISMATCH')
    table = StudyTable('MainContent_dvMethod')
    table.feed(html)
    description = ' '.join(cell for row in table.rows for cell in row if cell)
    if not description:
        raise ValueError('METHOD_DESCRIPTION_MISSING')
    return {'study_id': study_id, 'method_name': expected_name,
            'published_description': description,
            'source': f'https://npgsweb.ars-grin.gov/gringlobal/method.aspx?id={study_id}'}


def aggregate_studies(records: list[dict], excluded_studies: list[str]) -> list[dict]:
    """Equal study weights, first averaging within study across published inventory rows.

    Intended for authorized source reconstruction before partition freezing only.
    Never fit a label model, impute a value, or use a genotype/prediction to weight.
    """
    values = defaultdict(lambda: defaultdict(list))
    identities = defaultdict(set)
    for record in records:
        identity = record['accession_id']
        identities[identity].add(record['grin_accession_id'])
        if record['study_id'] in excluded_studies or not record['finite_numeric_value']:
            continue
        values[identity][record['study_id']].append(float(record['raw_value']))
    if any(len(ids) != 1 for ids in identities.values()):
        raise ValueError('NORMALIZED_ACCESSION_AMBIGUITY')
    results = []
    for identity, studies in sorted(values.items()):
        means = [math.fsum(v)/len(v) for v in studies.values()]
        results.append({'accession_id': identity,
                        'grin_accession_id': next(iter(identities[identity])),
                        'protein': math.fsum(means)/len(means),
                        'study_count': len(studies),
                        'published_value_count': sum(len(v) for v in studies.values()),
                        'study_ids': ';'.join(sorted(studies)),
                        'unit': '% seed dry weight'})
    return results
