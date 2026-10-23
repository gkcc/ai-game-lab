"""Exercise real official and Worker loaders on saved observations, no steps."""
import argparse
from copy import deepcopy
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from learning.bc_crop.audit import stable
from learning.task_bc_v1.deployment import JointAgent
from decision.audit_market_teacher import observation
from .entry import generate_wrapper


def _saved_inputs(previous):
    rows = read(previous / 'development-ledger.json')
    row = next(r for r in rows if r['seed'] == 31205138 and r['side'] == 0 and r['policy'] == 'trained')
    replay_path = previous / 'games' / row['game'] / 'replay.json.gz'
    assert sha(replay_path) == row['replay_gzip_sha256']
    with gzip.open(replay_path, 'rt', encoding='utf-8') as stream:
        replay = json.load(stream)
    return row, replay_path, replay


def _worker_child(out):
    """Import the actual frozen Worker with empty input; never dispatch play."""
    cfg = read(out / 'registration.json')
    old_stdin, saved_stdout = sys.stdin, os.dup(1)
    sys.stdin = io.StringIO('')
    worker = None
    try:
        spec = importlib.util.spec_from_file_location('entry_repair_frozen_worker', out / 'sources/worker.py')
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
    finally:
        sys.stdin = old_stdin
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
        if worker is not None and hasattr(worker, 'PROTOCOL'):
            worker.PROTOCOL.close()
    path = out / 'agents/regression-worker.py'
    actor, digest = worker.load_agent(str(path))
    assert digest == sha(path)
    assert actor.__globals__['__file__'] == str(path)
    _, _, replay = _saved_inputs(ROOT / cfg['previous_root'])
    actions = []
    for frame in replay['steps'][:2]:
        actions.append(actor(observation(frame, 0), deepcopy(replay['configuration'])))
    save(out / 'regression-local/worker-actions.json', actions)


def _trace(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 2
    assert all(not any(r[k] for k in ('fallbacks', 'load_errors', 'inference_errors')) for r in records)
    return stable(records)


def run(out):
    from .experiment import ready
    from kaggle_environments.agent import get_last_callable
    from kaggle_environments.errors import InvalidArgument
    import kaggle_environments.agent as official_agent

    out = Path(out).resolve()
    cfg = ready(out)
    with (out / 'regression-started.json').open('x', encoding='utf-8') as stream:
        stream.write('{}\n')
    previous = ROOT / cfg['previous_root']
    local = out / 'regression-local'
    local.mkdir(exist_ok=False)
    row, replay_path, replay = _saved_inputs(previous)
    old = previous / 'agents/entry-31205138-0-trained.py'
    official_path = generate_wrapper(out, 'regression-official', cfg)
    worker_path = generate_wrapper(out, 'regression-worker', cfg)
    direct_trace = out / 'traces/regression-direct.jsonl.gz'
    assert not direct_trace.exists()
    initial_cwd, initial_sys_path = Path.cwd(), list(sys.path)
    try:
        os.chdir(ROOT.parent)
        assert Path.cwd() != ROOT
        try:
            get_last_callable(old.read_text(encoding='utf-8'), path=str(old.resolve()))
        except InvalidArgument as exc:
            # Exact exception type plus underlying missing-variable diagnostic.
            text = str(exc)
            assert 'NameError' in text and "__file__" in text and 'not defined' in text
            old_failure = {'type': type(exc).__name__, 'cause': 'NameError: __file__ is not defined'}
        else:
            raise AssertionError('Old official filename entry unexpectedly loaded')
        # The official loader leaks its appended path on failure. Restore caller
        # paths before the independent successful loader invocation.
        sys.path[:] = initial_sys_path
        official = get_last_callable(official_path.read_text(encoding='utf-8'), path=str(official_path))
        assert '__file__' not in official.__globals__
        assert official is official.__globals__['submission_agent']
        assert official is official.__globals__['agent']
        direct = JointAgent(ROOT / cfg['crop_checkpoint'], cfg['crop_checkpoint_sha256'],
                            out / 'model/trained.npz', sha(out / 'model/trained.npz'), trace=direct_trace)
        actions = []
        for step, frame in enumerate(replay['steps'][:2]):
            a = official(observation(frame, 0), deepcopy(replay['configuration']))
            b = direct(observation(frame, 0), deepcopy(replay['configuration']))
            assert a == b == replay['steps'][step + 1][0]['action']
            actions.append(a)
        save(local / 'official-direct-actions.json', actions)
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT)
        env['KAGGRICULTURE_ROOT'] = str(ROOT)
        command = [sys.executable, '-B', '-m', 'learning.task_entry_repair_v1.verify_entry',
                   '--out', str(out), '--worker-child']
        try:
            child = subprocess.run(command, cwd=ROOT.parent, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=120,
                                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        except subprocess.TimeoutExpired as exc:
            # subprocess.run kills and waits for its child before raising.
            (local / 'worker-local.log').write_bytes((exc.stdout or b'') + (exc.stderr or b'') + b'\nWORKER_TIMEOUT\n')
            raise
        (local / 'worker-local.log').write_bytes(child.stdout + child.stderr)
        assert child.returncode == 0, 'Frozen Worker loader regression failed; see local log'
        assert read(local / 'worker-actions.json') == actions
        official_trace = out / 'traces/regression-official.jsonl.gz'
        worker_trace = out / 'traces/regression-worker.jsonl.gz'
        traces = _trace(official_trace)
        assert traces == _trace(worker_trace) == _trace(direct_trace)
        old_trace = previous / 'traces' / (row['game'] + '.jsonl.gz')
        assert sha(old_trace) == row['trace_sha256']
        with gzip.open(old_trace, 'rt', encoding='utf-8') as stream:
            old_turns = [json.loads(next(stream)), json.loads(next(stream))]
        assert traces == stable(old_turns)
        files = [old, official_path, worker_path, replay_path, old_trace, official_trace, worker_trace, direct_trace,
                 out / 'sources/worker.py', out / 'model/trained.npz', ROOT / cfg['crop_checkpoint']]
        receipt = dict(passed=True, old_official_counterexample=old_failure,
            freeze_sha256=sha(out / 'freeze.json'),
            registration_sha256=sha(out / 'registration.json'),
            official_loader_sha256=sha(Path(official_agent.__file__)),
            files_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files},
            observation_steps=[0, 1], seed=31205138, side=0,
            official_load_attempts=2, worker_load_attempts=1,
            official_agent_calls=2, worker_agent_calls=2, direct_calls=2,
            action_equalities=2, complete_stable_trace_equalities=2,
            equals_original_saved_actions_and_traces=True,
            worker_loader='Actual frozen module import with empty stdin, then load_agent; no play requests',
            official_loader='get_last_callable with absolute filename and no injected __file__',
            non_project_cwd=True, regression_trace_paths_separate=True,
            official_transitions=0, complete_games=0, training_updates=0)
        save(out / 'regressions.json', receipt)
        return receipt
    finally:
        os.chdir(initial_cwd)
        sys.path[:] = initial_sys_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--worker-child', action='store_true')
    args = parser.parse_args()
    if args.worker_child:
        _worker_child(args.out.resolve())
    else:
        run(args.out)
