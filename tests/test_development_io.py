import hashlib
import json
from pathlib import Path

import pytest

from fprior.io import load_development_phenotypes


def fixture(tmp_path):
    rows = [b'a\t40.5\n', b'b\t42.0\n']
    labels = tmp_path / 'development.tsv'
    labels.write_bytes(b''.join(rows))
    index = {}
    offset = 0
    for identity, row in zip(['a', 'b'], rows):
        index[identity] = dict(offset=offset, length=len(row),
                               sha256=hashlib.sha256(row).hexdigest())
        offset += len(row)
    (tmp_path / 'index.json').write_text(json.dumps(index))
    (tmp_path / 'sealed_ids.txt').write_text('sealed\n')
    records = {}
    for key, filename in [('labels', 'development.tsv'), ('index', 'index.json'),
                           ('sealed_ids', 'sealed_ids.txt')]:
        records[key] = dict(file=filename,
                            sha256=hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest())
    manifest = tmp_path / 'partition_manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, status='FROZEN',
                                       development_interface=records)))
    return manifest


def test_reads_requested_order_and_rejects_corrupted_row(tmp_path):
    manifest = fixture(tmp_path)
    assert load_development_phenotypes(manifest, ['b', 'a']).tolist() == [42, 40.5]
    (tmp_path / 'development.tsv').write_bytes(b'a\t99.5\nb\t42.0\n')
    with pytest.raises(ValueError, match='LABEL_ROW_MISMATCH'):
        load_development_phenotypes(manifest, ['a'])


def test_sealed_and_unknown_ids_fail_before_opening_labels(tmp_path, monkeypatch):
    manifest = fixture(tmp_path)
    original = Path.open

    def guard(self, *args, **kwargs):
        if self.name == 'development.tsv':
            raise AssertionError('labels opened before authorization')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guard)
    with pytest.raises(ValueError, match='SEALED_ID_ACCESS_FORBIDDEN'):
        load_development_phenotypes(manifest, ['sealed'])
    with pytest.raises(ValueError, match='ID_NOT_PERMITTED'):
        load_development_phenotypes(manifest, ['unknown'])
