"""Configuration preflight and numerical contracts; never reads source phenotypes."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tomllib
import csv
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
        'data': ['directory', 'source_manifest', 'raw_calls', 'sealed_partition_tsv',
                 'partition_manifest'],
        'regions': ['fasta', 'bed'], 'teacher': ['manifest'],
        'evaluation': ['outer_split_file', 'inner_split_file'],
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


def load_development_phenotypes(manifest_path: Path, sample_ids: list[str]) -> np.ndarray:
    """Validate permission metadata first, then seek only requested development rows.

    This interface never opens raw phenotype sources or the sealed-value artifact.
    The authenticated index contains IDs and byte offsets, without phenotype values.
    Per-row hashes and identity checks reject stale, reordered or modified data.
    """
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('status') != 'FROZEN' or manifest.get('schema_version') != 1:
        raise ValueError('PARTITION_NOT_FROZEN')
    directory = path.parent

    def artifact(name):
        record = manifest['development_interface'][name]
        target = (directory / record['file']).resolve()
        if target.parent != directory:
            raise ValueError('DEVELOPMENT_ARTIFACT_PATH_INVALID')
        return target, record

    sealed_path, sealed_record = artifact('sealed_ids')
    sealed_bytes = sealed_path.read_bytes()
    if hashlib.sha256(sealed_bytes).hexdigest() != sealed_record['sha256']:
        raise ValueError('SEALED_ID_MANIFEST_MISMATCH')
    sealed = frozenset(sealed_bytes.decode().splitlines())
    assert_development_only(sample_ids, sealed)
    index_path, index_record = artifact('index')
    index_bytes = index_path.read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != index_record['sha256']:
        raise ValueError('DEVELOPMENT_INDEX_MISMATCH')
    index = json.loads(index_bytes)
    if set(index) & sealed or not set(sample_ids) <= set(index):
        raise ValueError('DEVELOPMENT_ID_NOT_PERMITTED')
    labels_path, _ = artifact('labels')
    result = []
    with labels_path.open('rb') as stream:
        for identity in sample_ids:
            entry = index[identity]
            stream.seek(entry['offset'])
            row = stream.read(entry['length'])
            if hashlib.sha256(row).hexdigest() != entry['sha256']:
                raise ValueError('DEVELOPMENT_LABEL_ROW_MISMATCH')
            observed, value = row.decode().rstrip('\n').split('\t')
            if observed != identity:
                raise ValueError('DEVELOPMENT_LABEL_ID_MISMATCH')
            result.append(float(value))
    return finite_array(result, 1, 'development_phenotypes')


class OfficialDevelopmentData:
    """Metadata and genotype access to frozen development IDs, without outcomes."""

    def __init__(self, config, root):
        self.config, self.root = config, Path(root)
        self.directory = self.root/config['data']['directory']
        self.manifest_path = self.root/config['data']['partition_manifest']
        partition = json.loads(self.manifest_path.read_text())
        if partition.get('status') != 'FROZEN' or partition['partition_version'] != config['evaluation']['partition']['version']:
            raise ValueError('PARTITION_VERSION_MISMATCH')
        record = partition['development_interface']['sealed_ids']
        ids_path = (self.directory/record['file']).resolve()
        if ids_path.parent != self.directory.resolve():
            raise ValueError('SEALED_ID_PATH_INVALID')
        contents = ids_path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != record['sha256']:
            raise ValueError('SEALED_ID_MANIFEST_MISMATCH')
        self.sealed = frozenset(contents.decode().splitlines())
        for name, field in [('splits.tsv', 'outer_split_file'), ('nested_splits.tsv', 'inner_split_file')]:
            entry = partition['artifacts'][name]
            path = (self.directory/entry['file']).resolve()
            if path != (self.root/config['evaluation'][field]).resolve():
                raise ValueError('FROZEN_SPLIT_PATH_MISMATCH')
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('FROZEN_SPLIT_HASH_MISMATCH')
        source = json.loads((self.root/config['data']['source_manifest']).read_text())
        if (source['reference_build'] != config['data']['reference_build']
                or source['array_axes'] != ['variant', 'sample']
                or source['genotype_encoding'] != config['data']['genotype_encoding']):
            raise ValueError('GENOTYPE_SOURCE_CONTRACT_MISMATCH')
        # Validate only requested non-phenotype artifacts; never iterate/re-hash
        # the sealed-label or prefreeze phenotype artifact entries.
        wanted = {'teacher_panel_windows.json', 'teacher_panel_logits.json', 'teacher_manifest.json',
                  'cohort_ids.tsv', 'samples.tsv', 'variants.tsv', 'gt_audit.json', 'partition_manifest.json'}
        verified = set()
        for entry in source['artifacts']:
            name = Path(entry['path']).name
            if name in wanted:
                path = self.root/entry['path']
                if path.resolve().parent != self.directory.resolve():
                    raise ValueError('TEACHER_CACHE_PATH_INVALID')
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                    raise ValueError('TEACHER_CACHE_HASH_MISMATCH')
                verified.add(name)
        if verified != wanted:
            raise ValueError('TEACHER_CACHE_NOT_AUTHENTICATED')
        teacher = json.loads((self.root/config['teacher']['manifest']).read_text())
        if (self.root/config['teacher']['manifest']).resolve() != (self.directory/'teacher_manifest.json').resolve():
            raise ValueError('TEACHER_MANIFEST_PATH_MISMATCH')
        if (teacher['identity']['adapter_revision'] != config['teacher']['revision']
                or teacher['inference_version'] != config['teacher']['inference_version']
                or teacher['acceptance_status'] != 'TEACHER_LOGITS_ACCEPTED'
                or teacher['identity']['inference_batch_size'] != config['teacher']['inference_batch_size']
                or teacher['identity']['inference_batch_size'] != 1
                or teacher['identity']['label_mapping'] != {'negative': 0, 'positive': 1}
                or teacher['identity']['sequence_length'] != config['teacher']['sequence_length']
                or teacher['identity']['output_definition'] != config['teacher']['output']
                or teacher['identity']['dtype'] != config['teacher']['precision_reference']):
            raise ValueError('TEACHER_VERSION_MISMATCH')
        validation_path = (self.root/config['teacher']['manifest']).parent/teacher['validation_file']
        validation_bytes = validation_path.read_bytes()
        if hashlib.sha256(validation_bytes).hexdigest() != teacher['validation_sha256']:
            raise ValueError('TEACHER_VALIDATION_HASH_MISMATCH')
        validation = json.loads(validation_bytes)
        self.teacher_validation = validation
        audit = json.loads((self.directory/'gt_audit.json').read_text())
        if any('|' in key for key in audit['raw_gt_counts']):
            raise ValueError('PHASED_SOURCE_REQUIRES_EXPLICIT_PHASE_DESIGN')

        def rows(path):
            with Path(path).open() as stream:
                return list(csv.DictReader(stream, delimiter='\t'))
        cohort = rows(self.directory/'cohort_ids.tsv')
        self.sample_index = {r['accession_id']: int(r['vcf_sample_index']) for r in cohort}
        if len(self.sample_index) != len(cohort):
            raise ValueError('COHORT_ID_DUPLICATE')
        self.splits = rows(self.root/config['evaluation']['outer_split_file'])
        self.nested = rows(self.root/config['evaluation']['inner_split_file'])
        self.development = {r['accession_id']: r for r in self.splits if r['partition'] == 'development'}
        assert_development_only(list(self.development), self.sealed)
        if set(self.development) | self.sealed != set(self.sample_index):
            raise ValueError('PARTITION_COHORT_MISMATCH')
        self.dosages = np.load(self.directory/'dosages.npy', mmap_mode='r')
        if (self.dosages.shape != (audit['retained_variants'], audit['samples'])
                or self.dosages.dtype != np.dtype('float32')
                or max(self.sample_index.values()) >= self.dosages.shape[1]):
            raise ValueError('GENOTYPE_ARRAY_AXES_INVALID')
        panel = json.loads((self.directory/'teacher_panel_windows.json').read_text())
        logits = json.loads((self.directory/'teacher_panel_logits.json').read_text())
        self.windows = [[v['index'] for v in w['variants']] for w in panel]
        self.outputs = [np.asarray([float(np.float32(np.float32(logits[w['haplotype_sequences'][format(i, f'0{k}b')]][1])-
                         np.float32(logits[w['haplotype_sequences'][format(i, f'0{k}b')]][0])))
                         for i in range(1 << k)]) for w in panel for k in [len(w['variants'])]]

    def genotypes(self, sample_ids):
        assert_development_only(sample_ids, self.sealed)
        if not set(sample_ids) <= set(self.development):
            raise ValueError('DEVELOPMENT_ID_NOT_PERMITTED')
        return np.asarray(self.dosages[:, [self.sample_index[i] for i in sample_ids]].T, dtype=np.float64)

    def context(self, outer_fold, budget):
        if outer_fold not in range(self.config['evaluation']['outer_folds']):
            raise ValueError('OUTER_FOLD_INVALID')
        selected = [r for r in self.nested if int(r['outer_fold']) == outer_fold and int(r['budget']) == budget]
        train = [r['accession_id'] for r in selected]
        test = [i for i, r in self.development.items() if int(r['outer_fold']) == outer_fold]
        assert_development_only(train+test, self.sealed)
        if not train or not test or len(train) > budget:
            raise ValueError('BUDGET_SPLIT_INVALID')
        if any(i not in self.development or int(self.development[i]['outer_fold']) == outer_fold for i in train):
            raise ValueError('OUTER_TRAIN_TEST_LEAKAGE')
        groups = [self.development[i]['group_id'] for i in train]
        if set(groups) & {self.development[i]['group_id'] for i in test}:
            raise ValueError('OUTER_GROUP_LEAKAGE')
        return train, test, np.asarray([int(r['inner_fold']) for r in selected]), groups
