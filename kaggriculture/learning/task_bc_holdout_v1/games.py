"""Bounded official filename matches: two isolated processes, no retries."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

# This file is also launched by its absolute filename from outside the project.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def planned(cfg):
    return [dict(game=f'holdout-{seed}-{opponent}-{side}-{arm}', seed=seed,
                 opponent=opponent, side=side, policy=arm)
            for seed in cfg['test_seeds'] for opponent in cfg['opponents']
            for arm in cfg['arms'] for side in (0, 1)]


def wrapper(out, key, arm, cfg):
    out = Path(out).resolve()
    if Path(key).name != key or key in ('', '.', '..'):
        raise ValueError('Expected a single artifact key')
    if arm not in ('trained', 'teacher_task'):
        raise ValueError('Unknown frozen arm')
    trace, path = out / 'traces' / (key + '.jsonl.gz'), out / 'agents' / (key + '.py')
    if trace.exists() or path.exists():
        raise FileExistsError('Preserve existing wrapper and trace')
    model = out / 'model/trained.npz'
    assert sha(ROOT / cfg['crop_checkpoint']) == cfg['crop_checkpoint_sha256']
    assert sha(model) == cfg['task_checkpoint_sha256']
    code = ('from pathlib import Path\nimport sys\n'
        '_source_path=globals().get("__file__",sys._getframe().f_code.co_filename)\n'
        'ROOT=next(p for p in Path(_source_path).resolve().parents if (p/"learning/task_bc_v1/deployment.py").is_file())\n'
        'sys.path.insert(0,str(ROOT))\nfrom learning.task_bc_v1.deployment import JointAgent\n'
        '_runner=JointAgent(ROOT/' + repr(cfg['crop_checkpoint']) + ',' + repr(cfg['crop_checkpoint_sha256']))
    if arm == 'trained':
        code += ',ROOT/' + repr(model.relative_to(ROOT).as_posix()) + ',' + repr(cfg['task_checkpoint_sha256'])
    code += (',trace=ROOT/' + repr(trace.relative_to(ROOT).as_posix()) + ')\n'
             'def agent(observation,configuration):\n    return _runner(observation,configuration)\n'
             'submission_agent=agent\n')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(code)
    assert path.read_text(encoding='utf-8') == code
    return path


def installed_engine_sha():
    package = Path(next(iter(importlib.util.find_spec('kaggle_environments').submodule_search_locations)))
    return sha(package / 'envs/kaggriculture/kaggriculture.py')


def job(out, key):
    """One process runs the real filename CLI and its official offline audit."""
    from learning.task_bc_holdout_v1.experiment import verify
    out = Path(out).resolve()
    cfg = verify(out)
    plan = next(p for p in planned(cfg) if p['game'] == key)
    attempt = read(out / 'job-inputs' / (key + '.json'))
    for name, value in plan.items():
        assert attempt[name] == value
    bot, rival = out / 'agents' / (key + '.py'), out / 'sources' / (plan['opponent'] + '.py')
    assert sha(bot) == attempt['bot_sha256']
    assert sha(rival) == cfg['source_sha256'][rival.name] == attempt['rival_sha256']
    assert installed_engine_sha() == cfg['engine_sha256']
    assert sha(out / 'model/trained.npz') == cfg['task_checkpoint_sha256']
    assert sha(ROOT / cfg['crop_checkpoint']) == cfg['crop_checkpoint_sha256']
    assert not (out / 'games' / key).exists()
    # run_match.py still calls Kaggle's official filename loader. runpy avoids
    # an unowned long-running match grandchild beneath the watchdog process.
    command = [str(ROOT / 'run_match.py'), '--bot', str(bot), '--opponent', str(rival),
               '--seed', str(plan['seed']), '--side', str(plan['side']), '--name', key,
               '--output-root', str(out / 'games')]
    saved_argv, saved_cwd = sys.argv, Path.cwd()
    try:
        sys.argv = command
        os.chdir(ROOT.parent)
        try:
            runpy.run_path(str(ROOT / 'run_match.py'), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise RuntimeError('Official filename CLI failed') from exc
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)
    folder = out / 'games' / key
    result = read(folder / 'result.json')
    assert result['passed'] and result['done'] and result['statuses'] == ['DONE', 'DONE']
    assert result['recorded_states'] == 720 and result['action_turns'] == 719 and not result['agent_errors']
    assert result['engine_sha256'] == cfg['engine_sha256'] and result['upstream_commit'] == cfg['engine_commit']
    assert result['engine_matches_official_commit']
    assert result['agent_file_sha256'][str(bot)] == sha(bot) == attempt['bot_sha256']
    assert result['agent_file_sha256'][str(rival)] == sha(rival) == attempt['rival_sha256']
    assert result['seed'] == plan['seed'] and result['bot_side'] == plan['side']
    replay = read(folder / 'replay.json')
    assert replay['info']['seed'] == plan['seed']
    assert len(replay['steps']) == 720
    initial = [farm['money'] for farm in replay['steps'][0][0]['observation']['farms']]
    banks = [farm['money'] for farm in replay['steps'][-1][0]['observation']['farms']]
    assert banks == result['final_money'] == result['rewards']
    with (folder / 'replay.json.gz').open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1) as stream:
            stream.write((folder / 'replay.json').read_bytes())
    side = plan['side']
    margin = banks[side] - banks[1-side]
    row = dict(plan, banks=banks, initial_banks=initial, bank=banks[side], opponent_bank=banks[1-side],
               initial_bank=initial[side], opponent_initial_bank=initial[1-side],
               net_profit=banks[side]-initial[side], opponent_net_profit=banks[1-side]-initial[1-side],
               margin=margin, score=1.0 if margin > 0 else .5 if margin == 0 else 0.,
               states=720, action_turns=719, status=result['statuses'], passed=True,
               bot_sha256=sha(bot), rival_sha256=sha(rival), engine_sha256=result['engine_sha256'],
               engine_commit=result['upstream_commit'], package=result['package_version'],
               semantic_replay_sha256=result['semantic_replay_sha256'], semantic_replay_sha256_version=1,
               semantic_replay_sha256_v2=result['semantic_replay_sha256_v2'], audit_version=2,
               rewards=result['rewards'], replay_gzip_sha256=sha(folder / 'replay.json.gz'),
               agent_errors=0, agent_relative=bot.relative_to(ROOT).as_posix(),
               trace_sha256=sha(out / 'traces' / (key + '.jsonl.gz')),
               task_checkpoint_sha256=cfg['task_checkpoint_sha256'] if plan['policy'] == 'trained' else None,
               crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'], official_filename_entry=True)
    # Preserve game completion even if the subsequent audit fails.
    write_new(out / 'job-results' / (key + '.row.json'), row)
    from learning.task_bc_holdout_v1.audit import audit
    certificate = audit(out, row, cfg)
    verify(out)
    write_new(out / 'job-results' / (key + '.audit.json'), certificate)
    write_new(out / 'job-results' / (key + '.complete.json'),
              dict(game=key, passed=True, row_sha256=sha(out / 'job-results' / (key + '.row.json')),
                   certificate_sha256=sha(out / 'job-results' / (key + '.audit.json'))))


def terminate_owned(process):
    """The match and audit run in this owned process; kill and reap it."""
    if process.poll() is None:
        process.kill()
    process.wait()


def run(out):
    from learning.task_bc_holdout_v1.experiment import ready, verify
    out = Path(out).resolve()
    cfg = ready(out)
    gate = read(out / 'preflight.json')
    assert gate['passed'] and gate['freeze_sha256'] == sha(out / 'freeze.json')
    assert cfg['game_budget'] == 96
    assert cfg['max_workers'] in (1, 2)
    assert cfg['per_game_timeout_seconds'] == 900
    assert cfg['arms'] == ['trained', 'teacher_task']
    assert cfg['opponents'] == ['v1', 'market', 'round2']
    assert len(cfg['test_seeds']) == 8 and len(set(cfg['test_seeds'])) == 8
    assert installed_engine_sha() == cfg['engine_sha256']
    verify(out, history=True)
    write_new(out / 'games-started.json', dict(freeze_sha256=sha(out / 'freeze.json')))
    all_plan = planned(cfg)
    order = {p['game']: i for i, p in enumerate(all_plan)}
    ledger, certificates, attempts, failures = [], [], [], []
    active = {}
    severe = False
    started = time.monotonic()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

    def persist():
        ledger.sort(key=lambda r: order[r['game']])
        certificates.sort(key=lambda r: order[r['game']])
        save(out / 'ledger.json', ledger)
        save(out / 'audit.json', certificates)
        save(out / 'attempts.json', attempts)
        save(out / 'failures.json', failures)

    def launch(plan):
        verify(out)
        key = plan['game']
        bot = wrapper(out, key, plan['policy'], cfg)
        item = dict(plan, bot_sha256=sha(bot), rival_sha256=sha(out / 'sources' / (plan['opponent'] + '.py')))
        # Persist identity and consume the attempt before any official process.
        write_new(out / 'job-inputs' / (key + '.json'), item)
        attempts.append(item)
        persist()
        logpath = out / 'job-local' / (key + '.log')
        logpath.parent.mkdir(parents=True, exist_ok=True)
        log = logpath.open('xb')
        try:
            options = dict(creationflags=subprocess.CREATE_NO_WINDOW) if sys.platform == 'win32' else dict(start_new_session=True)
            process = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-B', '-X', 'utf8',
                                        str(Path(__file__).resolve()), 'job', '--out', str(out), '--key', key],
                                       cwd=ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT, **options)
        except BaseException:
            log.close()
            raise
        active[key] = dict(process=process, log=log, start=time.monotonic())

    persist()
    try:
        for seed in cfg['test_seeds']:
            queue = [p for p in all_plan if p['seed'] == seed]
            cursor = 0
            while active or (cursor < len(queue) and not failures):
                # Observe every existing process before dispatching replacements.
                for key, state in list(active.items()):
                    process = state['process']
                    timeout = process.poll() is None and time.monotonic()-state['start'] >= cfg['per_game_timeout_seconds']
                    if timeout:
                        terminate_owned(process)
                    code = process.poll()
                    if code is None:
                        continue
                    process.wait()
                    state['log'].close()
                    del active[key]
                    rowfile = out / 'job-results' / (key + '.row.json')
                    if rowfile.is_file():
                        ledger.append(read(rowfile))
                    marker = out / 'job-results' / (key + '.complete.json')
                    try:
                        assert not timeout and code == 0, 'Job timeout or exit failure'
                        receipt = read(marker)
                        certfile = out / 'job-results' / (key + '.audit.json')
                        assert receipt['passed'] and receipt['game'] == key
                        assert receipt['row_sha256'] == sha(rowfile)
                        assert receipt['certificate_sha256'] == sha(certfile)
                        certificates.append(read(certfile))
                    except Exception as exc:
                        failures.append(dict(game=key, timeout=timeout, returncode=code,
                                             error_type=type(exc).__name__, dispatch_stopped=True))
                    persist()
                    print(f'Completed official games {len(ledger)}/96; audited {len(certificates)}; failures {len(failures)}', flush=True)
                while not failures and cursor < len(queue) and len(active) < cfg['max_workers']:
                    launch(queue[cursor])
                    cursor += 1
                if active:
                    time.sleep(.1)
            if failures:
                break
            current = [r for r in ledger if r['seed'] == seed]
            assert len(current) == 12
            own = lambda arm: sum(r['bank'] for r in current if r['policy'] == arm)/6
            severe = own('trained') < .5 * own('teacher_task')
            if severe:
                save(out / 'severe-regression.json', dict(seed=seed, passed=False,
                     mean_trained_bank=own('trained'), mean_reference_bank=own('teacher_task'),
                     ratio_threshold=.5, dispatch_stopped=True))
                break
    except BaseException as exc:
        failures.append(dict(game=None, error_type=type(exc).__name__, dispatch_stopped=True))
        persist()
        raise
    finally:
        # Exceptional parent interruption must never orphan owned simulation jobs.
        for state in active.values():
            try:
                terminate_owned(state['process'])
            finally:
                state['log'].close()
        active.clear()
        save(out / 'workers.json', dict(closed=True, max_workers=cfg['max_workers']))
    verify(out, history=True)
    complete = len(ledger) == len(certificates) == 96 and not failures and not severe
    summary = dict(complete=complete, completed_games=len(ledger), audited_games=len(certificates),
                   attempts=len(attempts), failures=len(failures), severe_regression=severe,
                   official_reexecution_transitions=len(certificates)*719,
                   training_runs=0, wall_seconds=time.monotonic()-started,
                   independent_holdout=True, full_gate_evaluated=False)
    save(out / 'run-summary.json', summary)
    if complete:
        from learning.task_bc_holdout_v1.statistics import analyze
        try:
            save(out / 'statistics.json', analyze(ledger, cfg))
        except Exception as exc:
            summary['statistics_failure'] = dict(error_type=type(exc).__name__)
            save(out / 'run-summary.json', summary)
            raise
        summary['full_gate_evaluated'] = True
        summary['statistics_sha256'] = sha(out / 'statistics.json')
        save(out / 'run-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'job'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--key')
    args = parser.parse_args()
    if args.mode == 'run':
        summary = run(args.out)
        if not summary['complete']:
            raise SystemExit(1)
    else:
        if not args.key:
            parser.error('--key is required for job')
        try:
            job(args.out, args.key)
        except BaseException:
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
