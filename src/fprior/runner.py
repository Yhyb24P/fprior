"""Local development-only diagnostics, cost profiling and guarded evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time
import resource
import os
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from .io import OfficialDevelopmentData, load_development_phenotypes
from .io import assert_order
from .design import build_design
from .experiment import nested_predictions
from .eval import paired_metrics, group_bootstrap_interval


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


def _acceptance(config, root, data):
    if not config.get('primary_trait_id'):
        raise ValueError('TARGET_UNSPECIFIED')
    if config['primary_trait_id'] != 'protein':
        raise ValueError('TARGET_SOURCE_MISMATCH: frozen interface contains protein only')
    value = config['execution']['pretraining_acceptance']
    if not value:
        raise ValueError('PRETRAINING_ACCEPTANCE_MISSING: development Study audit and synthetic power validation required')
    evidence = json.loads((root/value).read_text())
    expected = hashlib.sha256(data.manifest_path.read_bytes()).hexdigest()
    if (evidence.get('experiment_version') != config['experiment_version']
            or evidence.get('partition_manifest_sha256') != expected
            or evidence.get('study_audit', {}).get('status') != 'PASSED'
            or evidence.get('study_audit', {}).get('development_only') is not True
            or evidence.get('synthetic_original', {}).get('datasets') != 720
            or evidence.get('synthetic_original', {}).get('status') != 'PASSED'
            or evidence.get('synthetic_sparse', {}).get('status') != 'PASSED'):
        raise ValueError('PRETRAINING_ACCEPTANCE_INCOMPLETE')


def run(command, config, root):
    data = OfficialDevelopmentData(config, root)
    if command == 'evaluate':
        _acceptance(config, root, data)  # before any labels are requested
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = root/config['output']['directory']/f"{config['experiment_version']}_{command}_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'config.json', config)
    write_json(directory/'inputs.json', {'partition_manifest_sha256': hashlib.sha256(data.manifest_path.read_bytes()).hexdigest(),
        'source_manifest_sha256': hashlib.sha256((root/config['data']['source_manifest']).read_bytes()).hexdigest(),
        'implementation_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/'src/fprior').glob('*.py'))},
        'control_seed_rule': config['controls']['replicate_seed_rule'],
        'equal_budget_hash_rule': 'SHA256(comma_join(SeedSequence(seed,1,repeat).generate_state(4)) + TAB + comma_join(weight:.17g))',
        'sealed_values_accessed': False, 'response': 'synthetic' if command == 'profile' else 'none' if command == 'diagnose' else 'development_protein'})
    start = time.perf_counter(); results = []; frozen = []; ids = []; groups = []; means = []
    ram_limit = config['budget']['max_host_ram_fraction']*os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
    budget = config['learning_curve']['primary_training_sample_budget']
    folds = [config['execution']['profile_outer_fold']] if command == 'profile' else range(config['evaluation']['outer_folds'])
    threshold = data.teacher_validation['signal_rms_threshold']
    build = lambda a, b: build_design(a, b, data.windows, data.outputs, config, noise_threshold=threshold)
    try:
        # Load estimator extensions before controlling their BLAS/OpenMP pools;
        # limits applied before a lazy import do not constrain newly loaded libs.
        from sklearn.linear_model import ElasticNet  # noqa: F401
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
        with threadpool_limits(limits=config['execution']['blas_threads']):
            write_json(directory/'threadpools.json', threadpool_info())
            for fold in folds:
                train_ids, test_ids, inner, train_groups = data.context(fold, budget)
                train = data.genotypes(train_ids); test = data.genotypes(test_ids)
                print(f'{command}: outer={fold}, train={len(train)}, test={len(test)}', flush=True)
                if command == 'diagnose':
                    contexts = []
                    # These are new functional-kernel diagnostics, not a rerun
                    # of the previously accepted raw-call audit.
                    for part in range(config['evaluation']['inner_folds']):
                        training = inner != part
                        design = build(train[training], train[~training])
                        contexts.append({'inner_fold': part, 'train_n': int(training.sum()),
                                         'diagnostics': design.diagnostics})
                    contexts.append({'inner_fold': None, 'train_n': len(train),
                                     'diagnostics': build(train, test).diagnostics})
                    record = {'outer_fold': fold, 'contexts': contexts, 'phenotypes_accessed': False}
                else:
                    if command == 'profile':
                        # Actual frozen genotype/split sizes; synthetic responses
                        # are deliberately unrelated to protein and its source.
                        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], 3, fold]))
                        y = rng.normal(size=len(train))
                    else:
                        y = load_development_phenotypes(data.manifest_path, train_ids)
                    def progress(value):
                        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                        write_json(directory/'progress.json', dict(value, outer_fold=fold, peak_host_rss_bytes=peak))
                        print(f"{command}: outer={fold}, inner={value['inner_fold']} completed ({value['seconds']:.1f}s)", flush=True)
                        if peak > ram_limit:
                            raise ValueError('COMPUTE_BUDGET_EXCEEDED: host RAM')
                    record = nested_predictions(train, test, y, inner, build, config,
                                                groups=train_groups, progress=progress)
                    record['outer_fold'] = fold; record['actual_training_samples'] = len(train)
                    prediction_path = directory/f'predictions_outer_{fold}.npz'
                    np.savez(prediction_path, sample_ids=np.asarray(test_ids), **record.pop('predictions'))
                    record['prediction_sha256'] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
                    frozen.append(prediction_path); ids.extend(test_ids)
                    groups.extend(data.development[i]['group_id'] for i in test_ids)
                    means.extend([record['training_mean']]*len(test_ids))
                print(f'{command}: outer={fold} completed', flush=True)
                write_json(directory/f'outer_{fold}.json', record)
                results.append({'outer_fold': fold, 'actual_training_samples': len(train)})
                del train, test
            if command == 'evaluate':
                # All outer predictions have been persisted and hashed before
                # outer values are requested by the evaluation interface.
                y = load_development_phenotypes(data.manifest_path, ids)
                predictions = {}; observed_ids = []
                for path in frozen:
                    with np.load(path, allow_pickle=False) as saved:
                        observed_ids.extend(saved['sample_ids'].tolist())
                        for family in saved.files:
                            if family != 'sample_ids':
                                predictions.setdefault(family, []).append(saved[family])
                predictions = {f: np.concatenate(p) for f, p in predictions.items()}
                assert_order(observed_ids, ids, 'OUTER_PREDICTION_ID')
                metrics = paired_metrics(y, predictions, train_means=means)
                bootstrap = {f: group_bootstrap_interval(y, p, predictions['M0'], groups,
                    replicates=config['evaluation']['descriptive_group_bootstrap_replicates'], seed=config['seed'])
                    for f, p in predictions.items() if f != 'M0'}
                contrasts = [('M1', 'MD'), ('M1', 'EN'), ('M2', 'EN')]
                contrasts += [('M1', f) for f in predictions if f.startswith('MP:')]
                contrasts += [('M2', f) for f in predictions if f in ('M1', 'MU', 'MW', 'MWG', 'MHap', 'MLocal', 'MI') or f.startswith('MRPerm:')]
                paired = {f'{a}_vs_{b}': group_bootstrap_interval(y, predictions[a], predictions[b], groups,
                    replicates=config['evaluation']['descriptive_group_bootstrap_replicates'], seed=config['seed'])
                    for a, b in contrasts}
                write_json(directory/'evaluation.json', {'metrics': metrics, 'group_bootstrap_vs_M0': bootstrap,
                    'critical_paired_contrasts': paired,
                    'interpretation': 'conditional descriptive intervals; shared training data make outer folds dependent',
                    'sealed_evaluation_authorized': False})
        status = {'status': 'COMPLETED', 'command': command, 'folds': results,
                  'seconds': time.perf_counter()-start, 'output': str(directory),
                  'phenotypes_accessed': command == 'evaluate', 'sealed_values_accessed': False,
                  'formal_comparison': command == 'evaluate'}
        status['peak_host_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        write_json(directory/'status.json', status)
        return status
    except Exception as exc:
        write_json(directory/'status.json', {'status': 'COMPARISON_INCOMPLETE', 'error': str(exc),
            'seconds': time.perf_counter()-start, 'sealed_values_accessed': False})
        raise
