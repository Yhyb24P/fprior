"""Recovery tests use real small nested fits and corrupt their saved evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import Future
import json
import shutil

import numpy as np
import pytest

import fprior.runner as runner
from fprior.design import build_design
from fprior.experiment import nested_predictions
from test_experiment import fixture


@pytest.fixture
def saved_profile(tmp_path, monkeypatch):
    config, q, windows, cubes = fixture()
    config['execution'].update(blas_threads=1, outer_workers=1, profile_response='development_protein', profile_outer_fold=0)
    config['follow_up']['training_budget'] = 128
    config['evaluation']['outer_folds'] = 2
    config['output']['directory'] = 'runs'
    config['data']['source_manifest'] = 'source_manifest.json'
    root = tmp_path/'project'; root.mkdir()
    source = root/'runs'/'profile'; source.mkdir(parents=True)
    shutil.copytree(Path(runner.__file__).parent, root/'src/fprior', ignore=shutil.ignore_patterns('__pycache__'))
    (root/'preprocessing').mkdir()
    shutil.copy2(Path(__file__).parents[1]/'preprocessing/dosages.py', root/'preprocessing/dosages.py')
    ids = [f'id_{i}' for i in range(64)]
    y = q[:, 0]-.5*q[:, 2]+.2*q[:, 3]*q[:, 4]
    def context(fold, budget):
        test = list(range(32, 64)) if fold == 0 else list(range(32))
        train = [i for i in range(64) if i not in test]
        return [ids[i] for i in train], [ids[i] for i in test], np.arange(32)%4, [f'g{i}' for i in train]
    data = SimpleNamespace(context=context, manifest_path=root/'partition.json',
        development={i: {'group_id': f'g{j}'} for j, i in enumerate(ids)})
    data.manifest_path.write_text('partition identity')
    (root/'source_manifest.json').write_text('source identity')
    requested = []
    def labels(path, requested_ids):
        requested.append(list(requested_ids))
        return np.array([y[ids.index(i)] for i in requested_ids])
    monkeypatch.setattr(runner, 'load_development_phenotypes', labels)
    build = lambda a, b: build_design(a, b, windows, cubes, config, noise_threshold=1e-8)
    def save_fold(directory, fold):
        train, test, inner, groups = context(fold, 128)
        indices = [ids.index(i) for i in train]; test_indices = [ids.index(i) for i in test]
        record = nested_predictions(q[indices], q[test_indices], y[indices], inner, build, config, groups=groups)
        split = directory/f'split_outer_{fold}.json'
        runner.write_json(split, dict(train_ids=train, test_ids=test, inner_folds=inner, train_groups=groups))
        pred = directory/f'predictions_outer_{fold}.npz'
        np.savez(pred, sample_ids=np.array(test), **record.pop('predictions'))
        record.update(outer_fold=fold, actual_training_samples=len(train), training_budget=128,
            split_sha256=runner.file_sha256(split), prediction_sha256=runner.file_sha256(pred))
        runner.write_json(directory/f'outer_{fold}.json', record)
        return dict(outer_fold=fold, actual_training_samples=len(train))
    save_fold(source, 0)
    runner.write_json(source/'config.json', config)
    inputs = dict(scientific_recipe_sha256=runner.recipe_sha256(config),
        partition_manifest_sha256=runner.file_sha256(data.manifest_path),
        source_manifest_sha256=runner.file_sha256(root/'source_manifest.json'),
        implementation_sha256=runner._implementation(root), response='development_protein',
        input_identity_verified=True, sealed_values_accessed=False,
        config_sha256=runner.file_sha256(source/'config.json'), limitations={})
    runner.write_json(source/'inputs.json', inputs)
    runner.write_json(source/'environment.json', runner._environment(config))
    shutil.copytree(root/'src/fprior', source/'implementation/src/fprior')
    shutil.copytree(root/'preprocessing', source/'implementation/preprocessing')
    runner.write_json(source/'status.json', dict(status='COMPLETED', command='profile', training_budget=128,
        sealed_values_accessed=False, folds=[dict(outer_fold=0, actual_training_samples=32)]))
    return SimpleNamespace(root=root, source=source, config=config, inputs=inputs, data=data,
                           labels=requested, save_fold=save_fold)


def test_profile_reuse_preserves_predictions_and_provenance(saved_profile):
    f = saved_profile; target = f.root/'runs'/'evaluate'; target.mkdir()
    runner._reuse_profile(f.source, target, f.config, f.inputs, f.root, f.data)
    assert runner.file_sha256(target/'predictions_outer_0.npz') == runner.file_sha256(f.source/'predictions_outer_0.npz')
    assert (target/'reused_profile'/'implementation/src/fprior/solver.py').exists()
    assert json.loads((target/'completion_outer_0.json').read_text())['origin'] == 'profile'
    train, test, _, _ = f.data.context(0, 128)
    assert f.labels and all(set(requested) <= set(train) for requested in f.labels)
    assert not any(set(requested) & set(test) for requested in f.labels)


@pytest.mark.parametrize('damage,message', [
    ('prediction_hash', 'PREDICTION_HASH'), ('test_order', 'PREDICTION_ID'),
    ('train_ids', 'FOLD_IDENTITY'), ('inner_ids', 'FOLD_IDENTITY'),
    ('missing_model', 'MODEL_PREDICTIONS'), ('missing_inner', 'INNER_FOLDS'),
    ('missing_candidate', 'CANDIDATES'), ('failed_EN', 'EN_NOT_CONVERGED'),
    ('large_gap', 'EN_NOT_CONVERGED'), ('missing_path', 'EN_PATH'),
    ('large_residual', 'NUMERICAL_FAILURE'), ('outer_labels', 'FOLD_NOT_COMPLETE'),
    ('selected_score', 'SELECTION'), ('selected_parameter', 'SEARCH_GRID'),
])
def test_damaged_fold_is_rejected_before_reuse(saved_profile, damage, message):
    f = saved_profile; p = f.source
    record = json.loads((p/'outer_0.json').read_text())
    if damage == 'prediction_hash':
        with (p/'predictions_outer_0.npz').open('ab') as stream:
            stream.write(b'corrupted')
    elif damage in ('test_order', 'missing_model'):
        with np.load(p/'predictions_outer_0.npz') as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        if damage == 'test_order': arrays['sample_ids'] = arrays['sample_ids'][::-1]
        else: arrays.pop('M2')
        np.savez(p/'predictions_outer_0.npz', **arrays)
        record['prediction_sha256'] = runner.file_sha256(p/'predictions_outer_0.npz')
    elif damage in ('train_ids', 'inner_ids'):
        split = json.loads((p/'split_outer_0.json').read_text())
        if damage == 'train_ids': split['train_ids'][0] = 'sealed_id'
        else: split['inner_folds'][0] = 3
        runner.write_json(p/'split_outer_0.json', split)
        record['split_sha256'] = runner.file_sha256(p/'split_outer_0.json')
    elif damage == 'missing_inner': record['inner_audit'].pop()
    elif damage == 'missing_candidate': record['inner_scores']['M2'].pop()
    elif damage == 'failed_EN': record['inner_audit'][0]['models']['EN']['fits'][0]['status'] = 'FAILED'
    elif damage == 'large_gap': record['inner_audit'][0]['models']['EN']['fits'][0]['dual_gap'] = 1.
    elif damage == 'missing_path': record['inner_audit'][0]['models']['EN']['fits'].pop()
    elif damage == 'large_residual': record['outer_audit']['selected_diagnostics']['M2']['relative_residual'] = .1
    elif damage == 'outer_labels': record['outer_labels_accessed'] = True
    elif damage == 'selected_score': record['selected']['M1']['mse'] += 1.
    elif damage == 'selected_parameter': record['selected']['EN']['parameters']['alpha_ratio'] = .123
    runner.write_json(p/'outer_0.json', record)
    with pytest.raises(ValueError, match=message):
        runner._verify_fold(p, 0, f.config, f.data)


@pytest.mark.parametrize('damage,message', [
    ('config', 'SCIENTIFIC_CONFIG'), ('inputs', 'INPUT_MISMATCH'),
    ('synthetic', 'INPUT_OR_RESPONSE'), ('source', 'COMPUTATION_SOURCE'),
    ('source_archive', 'SOURCE_ARCHIVE'), ('fold_function', 'FOLD_IMPLEMENTATION'),
    ('environment', 'SOFTWARE_VERSION'), ('unfinished', 'PROFILE_NOT_COMPLETED'),
    ('dosage_source', 'COMPUTATION_SOURCE'),
])
def test_incompatible_profile_is_rejected(saved_profile, damage, message):
    f = saved_profile; config = deepcopy(f.config); inputs = deepcopy(f.inputs)
    if damage == 'config': config['solver']['rho'] = [.5]
    elif damage == 'inputs': inputs['source_manifest_sha256'] = 'changed'
    elif damage == 'synthetic':
        prior = deepcopy(inputs); prior['response'] = 'synthetic'; runner.write_json(f.source/'inputs.json', prior)
    elif damage in ('source', 'fold_function', 'dosage_source'):
        name = 'solver.py' if damage == 'source' else 'runner.py'
        path = f.root/'preprocessing/dosages.py' if damage == 'dosage_source' else f.root/'src/fprior'/name
        text = path.read_text()
        if damage != 'fold_function': text += '\n# changed numerical source\n'
        else: text = text.replace('train, test = data.genotypes(train_ids)', 'train, test = data.genotypes(list(reversed(train_ids)))')
        path.write_text(text); inputs['implementation_sha256'] = runner._implementation(f.root)
    elif damage == 'source_archive': (f.source/'implementation/src/fprior/solver.py').write_text('corrupt')
    elif damage == 'environment':
        env = runner._environment(config); env['numpy'] = 'different'; runner.write_json(f.source/'environment.json', env)
    elif damage == 'unfinished':
        status = json.loads((f.source/'status.json').read_text()); status['status'] = 'RUNNING'
        runner.write_json(f.source/'status.json', status)
    with pytest.raises(ValueError, match=message):
        runner._verify_run_identity(f.source, config, inputs, f.root, profile=True)


def test_resume_runs_only_missing_fold_and_never_evaluates_partial_output(saved_profile, monkeypatch):
    f = saved_profile; target = f.root/'runs'/'evaluate'; target.mkdir()
    runner._reuse_profile(f.source, target, f.config, f.inputs, f.root, f.data)
    calls = []
    class InlinePool:
        def __init__(self, **kwargs): self._processes = {}
        def submit(self, function, command, fold, config, root, directory, implementation):
            calls.append(fold); future = Future()
            try: future.set_result(f.save_fold(directory, fold))
            except BaseException as exc: future.set_exception(exc)
            return future
        def shutdown(self, **kwargs): pass
    monkeypatch.setattr(runner, 'ProcessPoolExecutor', InlinePool)
    status = runner._execute('evaluate', f.config, f.root, target, f.data, f.inputs)
    assert status['status'] == 'COMPLETED' and calls == [1]
    assert [r['outer_fold'] for r in status['folds']] == [0, 1]
    calls.clear()
    runner._execute('evaluate', f.config, f.root, target, f.data, f.inputs)
    assert calls == []  # All folds are authenticated, not trained again.


def test_failure_keeps_completed_fold_and_refuses_partial_evaluation(saved_profile, monkeypatch):
    f = saved_profile; target = f.root/'runs'/'evaluate'; target.mkdir()
    runner._reuse_profile(f.source, target, f.config, f.inputs, f.root, f.data)
    class FailingPool:
        def __init__(self, **kwargs): self._processes = {}
        def submit(self, *args):
            future = Future(); future.set_exception(ValueError('BASELINE_NOT_CONVERGED')); return future
        def shutdown(self, **kwargs): pass
    monkeypatch.setattr(runner, 'ProcessPoolExecutor', FailingPool)
    with pytest.raises(ValueError, match='BASELINE_NOT_CONVERGED'):
        runner._execute('evaluate', f.config, f.root, target, f.data, f.inputs)
    assert (target/'completion_outer_0.json').exists()
    assert not (target/'evaluation.json').exists()
    assert json.loads((target/'status.json').read_text())['status'] == 'COMPARISON_INCOMPLETE'


def test_active_run_cannot_be_opened_twice(tmp_path):
    with runner._run_lock(tmp_path):
        with pytest.raises(ValueError, match='RUN_ALREADY_ACTIVE'):
            with runner._run_lock(tmp_path): pass


def test_evaluate_entry_reuses_profile_and_resumes_without_refitting(saved_profile, monkeypatch):
    f = saved_profile; calls = []
    monkeypatch.setattr(runner, 'OfficialDevelopmentData', lambda config, root: f.data)
    monkeypatch.setattr(runner, '_acceptance', lambda config, root, data: deepcopy(f.inputs))
    class InlinePool:
        def __init__(self, **kwargs): self._processes = {}
        def submit(self, function, command, fold, config, root, directory, implementation):
            calls.append(fold); future = Future()
            future.set_result(f.save_fold(directory, fold)); return future
        def shutdown(self, **kwargs): pass
    monkeypatch.setattr(runner, 'ProcessPoolExecutor', InlinePool)
    status = runner.run('evaluate', f.config, f.root, reuse_profile=f.source)
    target = Path(status['output'])
    assert status['status'] == 'COMPLETED' and calls == [1]
    assert json.loads((target/'inputs.json').read_text())['command'] == 'evaluate'
    assert (target/'implementation/preprocessing/dosages.py').is_file()
    calls.clear()
    status = runner.run('evaluate', f.config, f.root, resume=target)
    assert status['status'] == 'COMPLETED' and calls == []
    incompatible = deepcopy(f.config); incompatible['execution']['blas_threads'] = 2
    with pytest.raises(ValueError, match='RESUME_CONFIG_MISMATCH'):
        runner.run('evaluate', incompatible, f.root, resume=target)
    with (target/'predictions_outer_0.npz').open('rb') as stream:
        original = stream.read()
    (target/'predictions_outer_0.npz').chmod(0o644)
    (target/'predictions_outer_0.npz').write_bytes(original+b'corrupt')
    with pytest.raises(ValueError, match='PREDICTION_HASH'):
        runner.run('evaluate', f.config, f.root, resume=target)
    assert calls == []


def test_unarchived_dosage_transform_requires_matching_git_version(tmp_path):
    """The v10 compatibility case requires all historical source bytes to match."""
    import subprocess
    root = tmp_path
    (root/'src/fprior').mkdir(parents=True); (root/'preprocessing').mkdir()
    source = root/'src/fprior/solver.py'; source.write_text('original math\n')
    dosage = root/'preprocessing/dosages.py'; dosage.write_text('original dosage transform\n')
    for args in [('init', '-q'), ('add', '.'), ('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                                               'commit', '-q', '-m', 'Original source')]:
        subprocess.run(['git', *args], cwd=root, check=True)
    revision, content = runner._profile_source_revision(root, {'src/fprior/solver.py': runner.file_sha256(source)})
    assert revision and content == dosage.read_bytes()
    with pytest.raises(ValueError, match='SOURCE_VERSION_UNAVAILABLE'):
        runner._profile_source_revision(root, {'src/fprior/solver.py': 'unknown hash'})
