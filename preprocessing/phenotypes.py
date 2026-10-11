"""Parse official GRIN per-study values, retaining observation provenance."""
from html.parser import HTMLParser
import math
from pathlib import Path
import re


class StudyTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.cell = None
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == 'table' and dict(attrs).get('id') == 'searchtable':
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
