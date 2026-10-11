"""Configuration preflight and numerical contracts; never reads source phenotypes."""
from __future__ import annotations

from pathlib import Path
import tomllib
import numpy as np


def finite_array(values, ndim: int, name: str) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name}: complex values are not supported")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite {ndim}-dimensional data")
    return array


def load_config(path: Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    with path.open('rb') as stream:
        config = tomllib.load(stream)
    if config.get('schema_version') != 2:
        raise ValueError('CONFIG_INVALID: expected schema_version=2')
    return config, path.parent.parent


def validate_inputs(config: dict, root: Path) -> dict:
    """Check metadata only. Presence is insufficient to declare DATA_READY."""
    root = Path(root).resolve()
    target = config.get('primary_trait_id')
    if not isinstance(target, str) or not target.strip():
        return {'status': 'TARGET_UNSPECIFIED', 'issues': ['primary_trait_id'],
                'phenotypes_accessed': False, 'data_ready': False}
    file_fields = {
        'data': ['directory', 'source_manifest', 'raw_calls', 'sealed_partition_tsv'],
        'regions': ['fasta', 'bed'], 'teacher': ['manifest'],
        'evaluation': ['outer_split_file'],
    }
    issues = []
    for section, fields in file_fields.items():
        for field in fields:
            value = config.get(section, {}).get(field)
            key = f'{section}.{field}'
            if not isinstance(value, str) or not value.strip():
                issues.append({'field': key, 'reason': 'UNSPECIFIED'})
                continue
            path = root / value
            present = path.is_dir() if field == 'directory' else path.is_file()
            if not present:
                issues.append({'field': key, 'reason': 'MISSING'})
    for section, fields in {
        'data': ['reference_build', 'genotype_encoding', 'phenotype_unit'],
        'teacher': ['revision'],
    }.items():
        for field in fields:
            if not str(config.get(section, {}).get(field, '')).strip():
                issues.append({'field': f'{section}.{field}', 'reason': 'UNVERIFIED'})
    return {'status': 'MISSING_REQUIRED_INPUTS' if issues else 'INPUTS_PRESENT_UNVERIFIED',
            'issues': issues, 'primary_trait_id': config['primary_trait_id'],
            'phenotypes_accessed': False, 'data_ready': False,
            'experiment_version': config.get('experiment_version')}


def assert_development_only(sample_ids: list[str], sealed_ids: frozenset[str]) -> None:
    if any(not isinstance(value, str) or not value for value in sample_ids):
        raise ValueError('SAMPLE_ID_INVALID')
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError('SAMPLE_ID_DUPLICATE')
    if set(sample_ids) & sealed_ids:
        raise ValueError('SEALED_ID_ACCESS_FORBIDDEN')


def assert_order(observed: list[str], expected: list[str], kind: str) -> None:
    if observed != expected or len(set(observed)) != len(observed):
        raise ValueError(f'{kind}_ORDER_MISMATCH')
