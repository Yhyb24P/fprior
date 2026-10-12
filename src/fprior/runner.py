"""Development-only diagnostics, profiling and complete nested evaluation."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import multiprocessing
import os
import time
import resource
import traceback
import ast
import fcntl
import shutil
import subprocess
from contextlib import contextmanager
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from .io import OfficialDevelopmentData, load_development_phenotypes, assert_order, validate_inputs
from .design import build_design
from .experiment import nested_predictions
from .eval import development_summary


def write_json(path, value):
    def encode(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode, allow_nan=False)+'\n')
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def recipe_sha256(config):
    fields = ['seed', 'primary_trait_id', 'data', 'variant_policy', 'teacher', 'regions',
              'solver', 'evaluation', 'learning_curve', 'baselines', 'controls', 'models']
    if 'follow_up' in config:
        fields.append('follow_up')
    return hashlib.sha256(json.dumps({k: config[k] for k in fields}, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()



def training_budget(config):
    budget = config.get('follow_up', {}).get('training_budget',
                    config['learning_curve']['primary_training_sample_budget'])
    if (not isinstance(budget, int) or isinstance(budget, bool)
            or budget not in config['learning_curve']['training_sample_budgets']):
        raise ValueError('TRAINING_BUDGET_NOT_REGISTERED')
    if config['models']['primary_budget_only'] and budget != config['learning_curve']['primary_training_sample_budget']:
        raise ValueError('PRIMARY_BUDGET_ONLY')
    return budget

def _acceptance(config, root, data):
    """Check actual inputs and unchanged recipe; no methodological evidence gate."""
    if not config.get('primary_trait_id'):
        raise ValueError('TARGET_UNSPECIFIED')
    if config['primary_trait_id'] != 'protein':
        raise ValueError('TARGET_SOURCE_MISMATCH')
    preflight = validate_inputs(config, root)
    if preflight['issues']:
        raise ValueError('MISSING_REQUIRED_INPUTS: '+str(preflight['issues']))
    settings = config['development_evaluation']
    if recipe_sha256(config) != settings['locked_recipe_sha256']:
        raise ValueError('SCIENTIFIC_RECIPE_CHANGED')
    for key, path in [('partition_manifest_sha256', data.manifest_path),
                      ('source_manifest_sha256', root/config['data']['source_manifest']),
                      ('genotypes_sha256', data.directory/'dosages.npy')]:
        if file_sha256(path) != settings[key]:
            raise ValueError('INPUT_IDENTITY_MISMATCH: '+key)
    if len(data.development) != settings['expected_development_accessions'] or len(data.windows) != settings['expected_windows']:
        raise ValueError('DEVELOPMENT_SCOPE_MISMATCH')
    # Validate every fixed split before requesting any phenotype rows.
    groups = {}
    for row in data.development.values():
        groups.setdefault(row['group_id'], set()).add(int(row['outer_fold']))
    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError('OUTER_GROUP_LEAKAGE')
    for fold in range(config['evaluation']['outer_folds']):
        data.context(fold, training_budget(config))
    return dict(protocol_version=settings['protocol_version'], input_identity_verified=True,
                scientific_recipe_sha256=recipe_sha256(config), sealed_values_accessed=False,
                limitations={key: 'NOT_COMPLETED' for key in
                    ('study_observations', 'synthetic_original', 'synthetic_sparse')})


def _run_fold(command, fold, config, root, directory):
    root, directory = Path(root), Path(directory)
    try:
        data = OfficialDevelopmentData(config, root)
        train_ids, test_ids, inner, train_groups = data.context(fold, training_budget(config))
        write_json(directory/f'split_outer_{fold}.json', dict(train_ids=train_ids,
            test_ids=test_ids, inner_folds=inner, train_groups=train_groups))
        train, test = data.genotypes(train_ids), data.genotypes(test_ids)
        from sklearn.linear_model import ElasticNet  # noqa: F401
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
        if config['execution'].get('compute_device') == 'cuda':
            import torch
            torch.set_num_threads(config['execution']['blas_threads'])
        threshold = data.teacher_validation['signal_rms_threshold']
        build = lambda a, b: build_design(a, b, data.windows, data.outputs, config, noise_threshold=threshold)
        with threadpool_limits(limits=config['execution']['blas_threads']):
            write_json(directory/f'threadpools_outer_{fold}.json', threadpool_info())
            print(f'{command}: outer={fold}, train={len(train)}, test={len(test)}', flush=True)
            if command == 'diagnose':
                contexts = []
                for part in range(config['evaluation']['inner_folds']):
                    training = inner != part
                    contexts.append(dict(inner_fold=part, train_n=int(training.sum()),
                        diagnostics=build(train[training], train[~training]).diagnostics))
                contexts.append(dict(inner_fold=None, train_n=len(train), diagnostics=build(train, test).diagnostics))
                record = dict(outer_fold=fold, contexts=contexts, phenotypes_accessed=False)
            else:
                if command == 'profile' and config['execution'].get('profile_response', 'synthetic') != 'development_protein':
                    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], 3, fold]))
                    y = rng.normal(size=len(train))
                else:
                    y = load_development_phenotypes(data.manifest_path, train_ids)
                def progress(value):
                    write_json(directory/f'progress_outer_{fold}.json', dict(value, outer_fold=fold,
                        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
                    print(f"{command}: outer={fold}, inner={value['inner_fold']} completed ({value['seconds']:.1f}s)", flush=True)
                record = nested_predictions(train, test, y, inner, build, config,
                    groups=train_groups, progress=progress)
                prediction_path = directory/f'predictions_outer_{fold}.npz'
                np.savez(prediction_path, sample_ids=np.asarray(test_ids), **record.pop('predictions'))
                record['prediction_sha256'] = file_sha256(prediction_path)
            record.update(outer_fold=fold, actual_training_samples=len(train), training_budget=training_budget(config),
                split_sha256=file_sha256(directory/f'split_outer_{fold}.json'),
                peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            if config['execution'].get('compute_device') == 'cuda':
                record['gpu'] = dict(device=torch.cuda.get_device_name(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), dtype='float64')
            write_json(directory/f'outer_{fold}.json', record)
            print(f'{command}: outer={fold} completed', flush=True)
            return dict(outer_fold=fold, actual_training_samples=len(train))
    except Exception as exc:
        write_json(directory/f'failure_outer_{fold}.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


def _collect_predictions(data, config, directory):
    predictions, ids, groups, means, folds = {}, [], [], [], []
    expected = set(config['models']['families'])-{'MP', 'MRPerm', 'MRPerm3', 'MSign'}
    for prefix, count in [('MP', config['controls']['permutation_replicates']),
                          ('MRPerm', config['controls']['residual_weight_permutation_replicates']),
                          ('MRPerm3', config['controls']['residual_weight_permutation_replicates']),
                          ('M2Equal', config['controls']['equal_budget_replicates']),
                          ('MSign', config['controls'].get('sign_permutation_replicates', 0))]:
        expected.update(f'{prefix}:{i}' for i in range(count))
    expected.update(['AdditiveSelection', 'Deployment'])
    for fold in range(config['evaluation']['outer_folds']):
        record = json.loads((directory/f'outer_{fold}.json').read_text())
        path = directory/f'predictions_outer_{fold}.npz'
        if file_sha256(path) != record['prediction_sha256']:
            raise ValueError('PREDICTION_HASH_MISMATCH')
        _, test_ids, _, _ = data.context(fold, training_budget(config))
        with np.load(path, allow_pickle=False) as saved:
            assert_order(saved['sample_ids'].tolist(), test_ids, 'OUTER_PREDICTION_ID')
            if set(saved.files)-{'sample_ids'} != expected:
                raise ValueError('COMPARISON_INCOMPLETE: missing model predictions')
            for family in expected:
                if saved[family].shape != (len(test_ids),) or not np.isfinite(saved[family]).all():
                    raise ValueError('PREDICTION_INVALID')
                predictions.setdefault(family, []).append(saved[family])
        ids.extend(test_ids); groups.extend(data.development[i]['group_id'] for i in test_ids)
        means.extend([record['training_mean']]*len(test_ids)); folds.extend([fold]*len(test_ids))
    if len(set(ids)) != len(ids) or set(ids) != set(data.development):
        raise ValueError('DEVELOPMENT_COVERAGE_INCOMPLETE')
    return {f: np.concatenate(p) for f, p in sorted(predictions.items())}, ids, groups, means, folds


def _implementation(root):
    return {str(p.relative_to(root)): file_sha256(p)
            for p in sorted([*(root/'src/fprior').glob('*.py'), root/'preprocessing/dosages.py'])}


def _profile_source_revision(root, sources):
    """Recover the unarchived dosage transform from the matching Git version.

    The original v10 archive only listed src/fprior. A Git version must match
    every archived source before it can supply this additional numerical file.
    """
    revisions = subprocess.check_output(['git', 'log', '--format=%H', '--',
        'src/fprior', 'preprocessing/dosages.py'], cwd=root, text=True).splitlines()
    for revision in revisions:
        def content(name):
            return subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=root,
                                           stderr=subprocess.DEVNULL)
        try:
            if all(hashlib.sha256(content(name)).hexdigest() == digest for name, digest in sources.items()):
                return revision, content('preprocessing/dosages.py')
        except subprocess.CalledProcessError:
            continue
    raise ValueError('REUSE_NUMERICAL_SOURCE_VERSION_UNAVAILABLE')


def _environment(config):
    import platform
    import scipy
    import sklearn
    value = dict(python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__,
        cpu_available=len(os.sched_getaffinity(0)),
        cpu_model=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                        if line.startswith('model name')), 'unknown'),
        blas_threads=config['execution']['blas_threads'], outer_workers=config['execution'].get('outer_workers', 1))
    if config['execution'].get('compute_device') == 'cuda':
        import torch
        value.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    return value


@contextmanager
def _run_lock(directory):
    with (directory/'.run.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('RUN_ALREADY_ACTIVE') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _prediction_names(config):
    names = set(config['models']['families'])-{'MP', 'MRPerm', 'MRPerm3', 'MSign'}
    for prefix, count in [('MP', config['controls']['permutation_replicates']),
                         ('MRPerm', config['controls']['residual_weight_permutation_replicates']),
                         ('MRPerm3', config['controls']['residual_weight_permutation_replicates']),
                         ('M2Equal', config['controls']['equal_budget_replicates']),
                         ('MSign', config['controls'].get('sign_permutation_replicates', 0))]:
        names.update(f'{prefix}:{i}' for i in range(count))
    return names | {'AdditiveSelection', 'Deployment'}


def _verify_run_identity(directory, config, inputs, root, *, profile=False):
    saved = json.loads((directory/'inputs.json').read_text())
    prior = json.loads((directory/'config.json').read_text())
    if (file_sha256(directory/'config.json') != saved['config_sha256']
            or recipe_sha256(prior) != recipe_sha256(config)
            or saved.get('scientific_recipe_sha256') != inputs['scientific_recipe_sha256']):
        raise ValueError('REUSE_SCIENTIFIC_CONFIG_MISMATCH')
    for key in ('partition_manifest_sha256', 'source_manifest_sha256'):
        if saved.get(key) != inputs[key]:
            raise ValueError('REUSE_INPUT_MISMATCH: '+key)
    # The source manifest identifies teacher caches, while the original config
    # also locks the full dosage array and partition identity.
    if (prior['development_evaluation'] != config['development_evaluation']
            or prior['execution'].get('compute_device') != config['execution'].get('compute_device')
            or saved.get('response') != 'development_protein'
            or saved.get('input_identity_verified') is not True
            or saved.get('sealed_values_accessed') is not False):
        raise ValueError('REUSE_INPUT_OR_RESPONSE_MISMATCH')
    if not profile and prior != config:
        raise ValueError('RESUME_CONFIG_MISMATCH')
    if not profile and saved.get('command') != 'evaluate':
        raise ValueError('RESUME_REQUIRES_EVALUATE_DIRECTORY')
    old_sources, current = dict(saved['implementation_sha256']), inputs['implementation_sha256']
    historical_revision = None
    if profile and set(current)-set(old_sources) == {'preprocessing/dosages.py'} and set(old_sources) <= set(current):
        historical_revision, content = _profile_source_revision(root, old_sources)
        old_sources['preprocessing/dosages.py'] = hashlib.sha256(content).hexdigest()
    if set(old_sources) != set(current):
        raise ValueError('REUSE_SOURCE_CATALOG_MISMATCH')
    changed = []
    for name, digest in old_sources.items():
        archive = directory/'implementation'/name
        historical = name == 'preprocessing/dosages.py' and historical_revision is not None
        if (not archive.is_file() and not historical) or (archive.is_file() and file_sha256(archive) != digest):
            raise ValueError('REUSE_SOURCE_ARCHIVE_MISMATCH: '+name)
        if digest == current[name]:
            continue
        if not profile or name not in ('src/fprior/runner.py', 'src/fprior/__main__.py'):
            raise ValueError('REUSE_COMPUTATION_SOURCE_MISMATCH: '+name)
        if name.endswith('runner.py'):
            old_tree, new_tree = ast.parse(archive.read_text()), ast.parse((root/name).read_text())
            new_functions = {n.name: ast.dump(n, include_attributes=False)
                             for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            # Only run orchestration and added verification helpers may change.
            # All original fold computation, input, collection and serialization
            # functions remain identical, including their nested function bodies.
            for node in old_tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != 'run':
                    if new_functions.get(node.name) != ast.dump(node, include_attributes=False):
                        raise ValueError('REUSE_FOLD_IMPLEMENTATION_MISMATCH: '+node.name)
        changed.append(name)
    prior_env = json.loads((directory/'environment.json').read_text())
    current_env = _environment(config)
    for key in ('python', 'numpy', 'scipy', 'sklearn', 'torch', 'cuda'):
        if prior_env.get(key) != current_env.get(key):
            raise ValueError('REUSE_SOFTWARE_VERSION_MISMATCH: '+key)
    if profile:
        status = json.loads((directory/'status.json').read_text())
        rows = status.get('folds', [])
        if (status.get('status') != 'COMPLETED' or status.get('command') != 'profile'
                or status.get('sealed_values_accessed') is not False
                or prior['execution'].get('profile_response') != 'development_protein'
                or status.get('training_budget') != training_budget(config)
                or len(rows) != 1 or rows[0].get('outer_fold') != prior['execution']['profile_outer_fold']
                or not isinstance(rows[0].get('actual_training_samples'), int)):
            raise ValueError('REUSE_PROFILE_NOT_COMPLETED')
    return dict(source_inputs_sha256=file_sha256(directory/'inputs.json'),
                source_config_sha256=file_sha256(directory/'config.json'),
                source_implementation_sha256=old_sources,
                historical_source_revision=historical_revision,
                current_implementation_sha256=current, orchestration_changes=changed)


def _verify_fold(directory, fold, config, data):
    record_path = directory/f'outer_{fold}.json'
    record = json.loads(record_path.read_text())
    split_path, prediction_path = directory/f'split_outer_{fold}.json', directory/f'predictions_outer_{fold}.npz'
    train, test, inner, groups = data.context(fold, training_budget(config))
    split = dict(train_ids=train, test_ids=test, inner_folds=inner.tolist(), train_groups=groups)
    if (json.loads(split_path.read_text()) != split or file_sha256(split_path) != record['split_sha256']
            or record.get('outer_fold') != fold or record.get('actual_training_samples') != len(train)
            or record.get('training_budget') != training_budget(config)):
        raise ValueError('REUSE_FOLD_IDENTITY_MISMATCH')
    if record.get('status') != 'PREDICTIONS_FROZEN' or record.get('outer_labels_accessed') is not False:
        raise ValueError('REUSE_FOLD_NOT_COMPLETE')
    digest = file_sha256(prediction_path)
    if digest != record['prediction_sha256']:
        raise ValueError('REUSE_PREDICTION_HASH_MISMATCH')
    names = _prediction_names(config); fitted = names-{'AdditiveSelection', 'Deployment'}
    if set(record['selected']) != fitted or set(record['inner_scores']) != fitted:
        raise ValueError('REUSE_MODEL_CATALOG_MISMATCH')
    with np.load(prediction_path, allow_pickle=False) as saved:
        assert_order(saved['sample_ids'].tolist(), test, 'REUSE_PREDICTION_ID')
        if set(saved.files)-{'sample_ids'} != names:
            raise ValueError('REUSE_MODEL_PREDICTIONS_INCOMPLETE')
        for name in names:
            if saved[name].shape != (len(test),) or not np.isfinite(saved[name]).all():
                raise ValueError('REUSE_PREDICTION_INVALID')
        for alias, key, allowed in [('AdditiveSelection', 'additive_selection_family', {'M0', 'EN'}),
                                    ('Deployment', 'deployment_family', {'M0', 'EN', 'M1', 'M2', 'MU'})]:
            if record[key] not in allowed or not np.array_equal(saved[alias], saved[record[key]]):
                raise ValueError('REUSE_SELECTED_PREDICTION_MISMATCH')
    # Read only already-verified training IDs to check mean and convergence
    # thresholds. Outer evaluation labels stay unread until all folds exist.
    y = load_development_phenotypes(data.manifest_path, train)
    if not np.isfinite(record['training_mean']) or record['training_mean'] != float(y.mean()):
        raise ValueError('REUSE_TRAINING_LABEL_MISMATCH')
    from .baselines import elasticnet_grid, gbrt_grid
    en_grid = elasticnet_grid(config['baselines']['elasticnet'])
    def check_en(audit, target, expected):
        fits = audit.get('fits', [])
        limit = config['baselines']['elasticnet']['tol']*float(np.var(target))
        if [f.get('parameters') for f in fits] != expected:
            raise ValueError('REUSE_EN_PATH_INCOMPLETE')
        for fit in fits:
            gap = fit.get('dual_gap', 0. if fit.get('intercept_only') else np.nan)
            if (fit.get('status') != 'PASSED' or not np.isfinite(gap) or gap > limit*(1+1e-10)
                    or fit.get('iterations', 0) > config['baselines']['elasticnet']['max_iter']):
                raise ValueError('REUSE_EN_NOT_CONVERGED')
    audit = record['inner_audit']
    if [a['inner_fold'] for a in audit] != list(range(config['evaluation']['inner_folds'])):
        raise ValueError('REUSE_INNER_FOLDS_INCOMPLETE')
    for part, row in enumerate(audit):
        if (row['train_n'] != int(np.sum(inner != part)) or row['validation_n'] != int(np.sum(inner == part))
                or set(row['models'])-{'unique_weight_decompositions'} != fitted):
            raise ValueError('REUSE_INNER_AUDIT_INCOMPLETE')
        for name in fitted:
            scores = np.asarray(record['inner_scores'][name])
            if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
                    or (scores < 0).any() or row['models'][name]['candidate_count'] != len(scores)):
                raise ValueError('REUSE_CANDIDATES_INCOMPLETE')
        check_en(row['models']['EN'], y[inner != part], en_grid)
    outer = record['outer_audit']
    if (set(outer['selected_diagnostics']) != fitted
            or set(outer['models'])-{'unique_weight_decompositions'} != fitted):
        raise ValueError('REUSE_OUTER_AUDIT_INCOMPLETE')
    for name, selected in record['selected'].items():
        scores = record['inner_scores'][name]
        index = selected.get('config_index')
        tolerance = config['solver']['selection_relative_tolerance']*max(1., min(scores))
        if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(scores)
                or selected.get('mse') != scores[index] or scores[index] > min(scores)+tolerance
                or outer['models'][name]['candidate_count'] != len(scores)):
            raise ValueError('REUSE_SELECTION_MISMATCH: '+name)
        if name in ('EN', 'FU-GBRT'):
            grid = en_grid if name == 'EN' else gbrt_grid(config['baselines']['gbrt'])
            first = next(i for i, value in enumerate(scores) if value <= min(scores)+tolerance)
            if len(scores) != len(grid) or index != first or selected['parameters'] != grid[index]:
                raise ValueError('REUSE_SEARCH_GRID_MISMATCH: '+name)
        else:
            weights = np.asarray(selected['weights'], dtype=float)
            if (weights.ndim != 1 or not np.isfinite(weights).all() or (weights < 0).any()
                    or not (np.isclose(weights.sum(), 1.) or (weights == 0).all())
                    or not np.allclose(weights*config['solver']['simplex_denominator'],
                                       np.round(weights*config['solver']['simplex_denominator']))
                    or selected['rho'] != config['solver']['rho'][index % len(config['solver']['rho'])]):
                raise ValueError('REUSE_SEARCH_GRID_MISMATCH: '+name)
    deployment_order = [n for n in config['models']['families'] if n in ('M0', 'M1', 'M2', 'MU', 'EN')]
    for key, order in [('deployment_family', deployment_order),
                       ('additive_selection_family', [n for n in deployment_order if n in ('M0', 'EN')])]:
        best = min(record['selected'][n]['mse'] for n in order)
        tolerance = config['solver']['selection_relative_tolerance']*max(1., best)
        chosen = next(n for n in order if record['selected'][n]['mse'] <= best+tolerance)
        if record[key] != chosen:
            raise ValueError('REUSE_SELECTION_MISMATCH: '+key)
    parameter = record['selected']['EN']['parameters']
    path = [p for p in en_grid if p['l1_ratio'] == parameter['l1_ratio']
            and p['alpha_ratio'] >= parameter['alpha_ratio']]
    check_en(outer['models']['EN'], y, path)
    for name, diagnostic in outer['selected_diagnostics'].items():
        if name in ('EN', 'FU-GBRT'):
            if diagnostic.get('status') != 'PASSED':
                raise ValueError('REUSE_BASELINE_NOT_CONVERGED')
        else:
            residual = diagnostic.get('relative_residual', np.nan)
            if not np.isfinite(residual) or residual > config['solver']['relative_residual_max']:
                raise ValueError('REUSE_NUMERICAL_FAILURE')
    evidence = dict(outer_fold=fold, actual_training_samples=len(train), prediction_sha256=digest,
                    split_sha256=file_sha256(split_path), outer_record_sha256=file_sha256(record_path))
    certificate = directory/f'completion_outer_{fold}.json'
    if certificate.exists():
        prior = json.loads(certificate.read_text())
        if any(prior.get(k) != v for k, v in evidence.items()):
            raise ValueError('RESUME_FOLD_EVIDENCE_CHANGED')
    return evidence


def _reuse_profile(source, directory, config, inputs, root, data):
    proof = _verify_run_identity(source, config, inputs, root, profile=True)
    prior = json.loads((source/'config.json').read_text())
    fold = prior['execution']['profile_outer_fold']
    evidence = _verify_fold(source, fold, config, data)
    status = json.loads((source/'status.json').read_text())
    if status['folds'][0]['actual_training_samples'] != evidence['actual_training_samples']:
        raise ValueError('REUSE_PROFILE_SAMPLE_COUNT_MISMATCH')
    archive = directory/'reused_profile'
    archive.mkdir()
    for name in ('config.json', 'inputs.json', 'environment.json', 'status.json'):
        shutil.copy2(source/name, archive/name)
    shutil.copytree(source/'implementation', archive/'implementation')
    if proof['historical_source_revision'] is not None:
        path = archive/'implementation/preprocessing/dosages.py'
        path.parent.mkdir(parents=True)
        path.write_bytes(subprocess.check_output(['git', 'show',
            proof['historical_source_revision']+':preprocessing/dosages.py'], cwd=root))
    for name in (f'outer_{fold}.json', f'split_outer_{fold}.json', f'predictions_outer_{fold}.npz'):
        shutil.copy2(source/name, archive/name)
    if (_verify_run_identity(archive, config, inputs, root, profile=True) != proof
            or _verify_fold(archive, fold, config, data) != evidence):
        raise ValueError('REUSE_SOURCE_CHANGED_DURING_COPY')
    write_json(directory/'profile_reuse.json', dict(proof, source_directory=str(source),
        source_archive='reused_profile', evidence=evidence))
    # Publish the outer record last: recovery never mistakes a partial copy for
    # a completed fold. The source archive and provenance are already durable.
    for name in (f'split_outer_{fold}.json', f'predictions_outer_{fold}.npz', f'outer_{fold}.json'):
        shutil.copy2(archive/name, directory/name)
    if _verify_fold(directory, fold, config, data) != evidence:
        raise ValueError('REUSE_SOURCE_CHANGED_DURING_COPY')
    write_json(directory/f'completion_outer_{fold}.json', dict(evidence, origin='profile',
        source_inputs_sha256=proof['source_inputs_sha256']))
    (directory/f'predictions_outer_{fold}.npz').chmod(0o444)


def _run_checked_fold(command, fold, config, root, directory, implementation):
    if _implementation(Path(root)) != implementation:
        raise ValueError('RUN_IMPLEMENTATION_CHANGED')
    result = _run_fold(command, fold, config, root, directory)
    if _implementation(Path(root)) != implementation:
        raise ValueError('RUN_IMPLEMENTATION_CHANGED')
    return result


def run(command, config, root, *, resume=None, reuse_profile=None):
    root = Path(root)
    if (resume is not None or reuse_profile is not None) and command != 'evaluate':
        raise ValueError('RECOVERY_REQUIRES_EVALUATE')
    if resume is not None and reuse_profile is not None:
        raise ValueError('RESUME_AND_PROFILE_REUSE_ARE_EXCLUSIVE')
    data = OfficialDevelopmentData(config, root)
    uses_protein = (command == 'evaluate' or (command == 'profile' and
                    config['execution'].get('profile_response') == 'development_protein'))
    inputs = _acceptance(config, root, data) if uses_protein else {}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = (Path(resume).resolve() if resume is not None else
                 root/config['output']['directory']/f"{config['experiment_version']}_{command}_{stamp}")
    if resume is not None and directory.parent.resolve() != (root/config['output']['directory']).resolve():
        raise ValueError('RESUME_OUTPUT_DIRECTORY_INVALID')
    if resume is None:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory/'config.json', config)
    inputs.update(partition_manifest_sha256=file_sha256(data.manifest_path),
        source_manifest_sha256=file_sha256(root/config['data']['source_manifest']),
        implementation_sha256=_implementation(root),
        config_sha256=file_sha256(directory/'config.json'), command=command, sealed_values_accessed=False,
        response='development_protein' if uses_protein else 'synthetic' if command == 'profile' else 'none')
    if (root/'.git').exists():
        inputs['implementation_git_revision'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    with _run_lock(directory):
        if resume is not None:
            _verify_run_identity(directory, config, inputs, root)
            if (directory/'profile_reuse.json').exists():
                proof = json.loads((directory/'profile_reuse.json').read_text())
                checked = _verify_run_identity(directory/'reused_profile', config, inputs, root, profile=True)
                if checked != {k: proof[k] for k in checked}:
                    raise ValueError('RESUME_PROFILE_PROVENANCE_CHANGED')
                checked_fold = _verify_fold(directory/'reused_profile', proof['evidence']['outer_fold'], config, data)
                if checked_fold != proof['evidence']:
                    raise ValueError('RESUME_PROFILE_EVIDENCE_CHANGED')
        else:
            write_json(directory/'environment.json', _environment(config))
            write_json(directory/'inputs.json', inputs)
            for name in inputs['implementation_sha256']:
                path = directory/'implementation'/name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root/name, path)
        print(f'output: {directory}', flush=True)
        return _execute(command, config, root, directory, data, inputs, reuse_profile=reuse_profile)


def _execute(command, config, root, directory, data, inputs, *, reuse_profile=None):
    start, results = time.perf_counter(), []
    uses_protein = (command == 'evaluate' or (command == 'profile' and
                    config['execution'].get('profile_response') == 'development_protein'))
    folds = [config['execution']['profile_outer_fold']] if command == 'profile' else list(range(config['evaluation']['outer_folds']))
    workers = min(config['execution'].get('outer_workers', 1), len(folds))
    try:
        if reuse_profile is not None:
            _reuse_profile(Path(reuse_profile).resolve(), directory, config, inputs, root, data)
        if command == 'evaluate':
            for fold in folds:
                if (directory/f'outer_{fold}.json').exists():
                    evidence = _verify_fold(directory, fold, config, data)
                    results.append(dict(outer_fold=fold, actual_training_samples=evidence['actual_training_samples']))
                    certificate = directory/f'completion_outer_{fold}.json'
                    if not certificate.exists():
                        write_json(certificate, dict(evidence, origin='evaluate'))
                    (directory/f'predictions_outer_{fold}.npz').chmod(0o444)
        completed = {r['outer_fold'] for r in results}
        pending = [fold for fold in folds if fold not in completed]
        write_json(directory/'progress.json', dict(completed_outer_folds=sorted(completed), pending_outer_folds=pending))
        write_json(directory/'status.json', dict(status='RUNNING', command=command,
            completed_outer_folds=sorted(completed), pending_outer_folds=pending, sealed_values_accessed=False))
        if pending:
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
            todo = iter(pending)
            futures = {}
            def submit(fold):
                futures[pool.submit(_run_checked_fold, command, fold, config, root, directory,
                                    inputs['implementation_sha256'])] = fold
            try:
                for fold in list(pending)[:workers]:
                    next(todo); submit(fold)
                while futures:
                    finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in finished:
                        futures.pop(future)
                        result = future.result()
                        if command == 'evaluate':
                            evidence = _verify_fold(directory, result['outer_fold'], config, data)
                            write_json(directory/f"completion_outer_{result['outer_fold']}.json", dict(evidence, origin='evaluate'))
                            (directory/f"predictions_outer_{result['outer_fold']}.npz").chmod(0o444)
                        results.append(result)
                        completed.add(result['outer_fold'])
                        write_json(directory/'progress.json', dict(completed_outer_folds=sorted(completed),
                            pending_outer_folds=[f for f in folds if f not in completed]))
                    for _ in finished:
                        next_fold = next(todo, None)
                        if next_fold is not None:
                            submit(next_fold)
            except BaseException:
                # No new folds start after a concrete failure or interruption.
                # Finished fold artifacts survive; interrupted ones are recomputed.
                for process in list((pool._processes or {}).values()):
                    if process.is_alive():
                        process.terminate()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
        if command == 'evaluate':
            predictions, ids, groups, means, fold_ids = _collect_predictions(data, config, directory)
            # Every fold/model prediction is complete, persisted and authenticated
            # before evaluation requests its observed development values.
            y = load_development_phenotypes(data.manifest_path, ids)
            observed_path = directory/'evaluation_development.npz'
            np.savez(observed_path, sample_ids=np.asarray(ids), observed=y,
                groups=np.asarray(groups), outer_folds=fold_ids, train_means=means)
            with threadpool_limits(limits=workers*config['execution']['blas_threads']):
                summary = development_summary(y, predictions, fold_ids, groups, means, config)
            summary.update(limitations=inputs['limitations'], observed_sha256=file_sha256(observed_path),
                interpretation='conditional descriptive intervals; outer folds share training data; development only')
            write_json(directory/'evaluation.json', summary)
        status = dict(status='COMPLETED', command=command, folds=sorted(results, key=lambda r: r['outer_fold']),
            seconds=time.perf_counter()-start, output=str(directory), outer_workers=workers, training_budget=training_budget(config),
            blas_threads_per_worker=config['execution']['blas_threads'],
            phenotypes_accessed=uses_protein, sealed_values_accessed=False, independent_validation=False)
        write_json(directory/'status.json', status)
        return status
    except BaseException as exc:
        write_json(directory/'status.json', dict(status='COMPARISON_INCOMPLETE',
            command=command, error=str(exc) or type(exc).__name__,
            seconds=time.perf_counter()-start, sealed_values_accessed=False))
        raise
