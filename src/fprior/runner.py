"""Development-only diagnostics, profiling and complete nested evaluation."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import multiprocessing
import os
import time
import resource
import traceback
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


def run(command, config, root):
    data = OfficialDevelopmentData(config, root)
    uses_protein = (command == 'evaluate' or (command == 'profile' and
                    config['execution'].get('profile_response') == 'development_protein'))
    inputs = _acceptance(config, root, data) if uses_protein else {}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = root/config['output']['directory']/f"{config['experiment_version']}_{command}_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'config.json', config)
    inputs.update(partition_manifest_sha256=file_sha256(data.manifest_path),
        source_manifest_sha256=file_sha256(root/config['data']['source_manifest']),
        implementation_sha256={str(p.relative_to(root)): file_sha256(p)
            for p in sorted((root/'src/fprior').glob('*.py'))},
        config_sha256=file_sha256(directory/'config.json'), sealed_values_accessed=False,
        response='development_protein' if uses_protein else 'synthetic' if command == 'profile' else 'none')
    import platform
    import scipy
    import sklearn
    environment = dict(python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__,
        cpu_available=len(os.sched_getaffinity(0)),
        cpu_model=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                        if line.startswith('model name')), 'unknown'),
        blas_threads=config['execution']['blas_threads'], outer_workers=config['execution'].get('outer_workers', 1))
    if config['execution'].get('compute_device') == 'cuda':
        import torch
        environment.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    write_json(directory/'environment.json', environment)
    write_json(directory/'inputs.json', inputs)
    print(f'output: {directory}', flush=True)
    start, results = time.perf_counter(), []
    folds = [config['execution']['profile_outer_fold']] if command == 'profile' else list(range(config['evaluation']['outer_folds']))
    try:
        workers = min(config['execution'].get('outer_workers', 1), len(folds))
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(_run_fold, command, fold, config, root, directory) for fold in folds]
            for future in as_completed(futures):
                results.append(future.result())
                write_json(directory/'progress.json', dict(completed_outer_folds=sorted(r['outer_fold'] for r in results)))
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
    except Exception as exc:
        write_json(directory/'status.json', dict(status='COMPARISON_INCOMPLETE', error=str(exc),
            seconds=time.perf_counter()-start, sealed_values_accessed=False))
        raise
