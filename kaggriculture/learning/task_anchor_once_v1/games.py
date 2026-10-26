"""Actual official filename games, frozen-action reconstruction and replay."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import contextlib
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.audit import stable, load
from learning.task_bc_v1.deployment import JointAgent
from learning.task_bc_holdout_v1.games import write_new
from learning.task_request_gate_v1.games import bound_original, model_counts
from learning.task_request_gate_v1.policy import same_request
from learning.task_dagger_once_v1.policy import BoundedAgent
from learning.task_anchor_once_v1.experiment import read, verify, model_identity
from learning.task_anchor_once_v1.storage import owned_writes
from decision.audit_market_teacher import observation
from decision.full_audit import reexecute
from decision.semantics import canonical


def official_entry():
    """Same frozen CLI main; native output already enters the bounded parent pipe."""
    import run_match
    assert Path(run_match.__file__).resolve() == (ROOT/'run_match.py').resolve()
    previous = run_match.native_output
    try:
        run_match.native_output = lambda log: contextlib.nullcontext()
        return run_match.main()
    finally:
        run_match.native_output = previous


def job(out,key):
    out=Path(out).resolve();cfg=verify(out)
    budget=owned_writes(out,cfg['storage']['game_write_bytes'],
        allowed_prefixes=['games/'+key],
        allowed_paths=['traces/'+key+'.jsonl.gz',
            *['job-results/'+key+suffix for suffix in ('.row.json','.audit.json','.complete.json')]])
    error=None
    try:
        with budget:_job(out,key)
    except BaseException as exc:
        error=type(exc).__name__
        raise
    finally:
        receipt='job-results/'+key+'.storage.json'
        with owned_writes(out,cfg['storage']['small_receipt_bytes'],allowed_paths=[receipt]):
            write_new(out/receipt,dict(game=key,passed=error is None,error_kind=error,
                output=budget.snapshot(),receipt_accounted_by='reserved parent bookkeeping',
                native_output_accounted_by='bounded parent process log'))


def task_identity(out, plan, cfg):
    if plan['policy'] == 'teacher_task':
        return None, None
    if plan['policy'] == 'baseline':
        return ROOT/cfg['task_checkpoint'], cfg['task_checkpoint_sha256']
    if plan['policy'] == 'candidate':
        return Path(out)/'model/trained.npz', model_identity(out)
    raise ValueError('Unknown policy arm')


def actor(out, plan, cfg):
    path, digest = task_identity(out,plan,cfg)
    return BoundedAgent(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'],
                        path,digest,limits=cfg['game_limits'])


def wrapper(out, plan, cfg):
    out = Path(out)
    key = plan['game']
    path, trace = out/'agents'/(key+'.py'), out/'traces'/(key+'.jsonl.gz')
    assert Path(key).name == key and not path.exists() and not trace.exists()
    task, digest = task_identity(out,plan,cfg)
    task_arg = 'None' if task is None else 'ROOT/'+repr(task.relative_to(ROOT).as_posix())
    code = ('from pathlib import Path\nimport sys\n'
        '_p=globals().get("__file__",sys._getframe().f_code.co_filename)\n'
        'ROOT=next(p for p in Path(_p).resolve().parents if (p/"learning/task_dagger_once_v1/policy.py").is_file())\n'
        'sys.path.insert(0,str(ROOT))\nfrom learning.task_dagger_once_v1.policy import BoundedAgent\n'
        '_runner=BoundedAgent(ROOT/'+repr(cfg['crop_checkpoint'])+','+repr(cfg['crop_checkpoint_sha256'])+
        ','+task_arg+','+repr(digest)+',trace=ROOT/'+repr(trace.relative_to(ROOT).as_posix())+
        ',limits='+repr(cfg['game_limits'])+')\n'
        'def agent(observation,configuration):\n    return _runner(observation,configuration)\nsubmission_agent=agent\n')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(code)
    return path


def audit(out, row, cfg):
    out = Path(out)
    side, key = row['side'], row['game']
    rp, tp = out/'games'/key/'replay.json.gz', out/'traces'/(key+'.jsonl.gz')
    assert sha(rp) == row['replay_gzip_sha256'] and sha(tp) == row['trace_sha256']
    with gzip.open(rp,'rt',encoding='utf-8') as stream:
        replay = json.load(stream)
    agent, rival = actor(out,row,cfg), load(out/'sources'/(row['opponent']+'.py'))
    original = None
    if row['parity_kind'] == 'student':
        original = JointAgent(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'],
                              ROOT/cfg['task_checkpoint'],cfg['task_checkpoint_sha256'])
    else:
        assert row['parity_kind'] is None
    parity = Counter()
    parity_budget = bound_original(original,True,cfg['game_limits']) if original else None
    with gzip.open(tp,'rt',encoding='utf-8') as trace:
        for step,(before,after) in enumerate(zip(replay['steps'],replay['steps'][1:])):
            line = trace.readline()
            assert line
            stored = json.loads(line)
            obs = observation(before,side)
            request = agent(obs,replay['configuration'])
            assert same_request(request,after[side]['action'])
            assert same_request(rival(observation(before,1-side),replay['configuration']),after[1-side]['action'])
            assert stable(agent.last) == stable(stored), 'Reconstructed current labels/requests/counts differ'
            assert all(agent.last[k] == 0 for k in ('fallbacks','load_errors','inference_errors'))
            if original is not None:
                parity_budget.begin_turn(); parity_budget.controller()
                assert same_request(original(obs,replay['configuration']),request)
                parity.update(model_counts(original.last,True))
                assert dict(parity) == parity_budget.counts
        assert not trace.readline()
    execution = Counter()
    def sink(event):
        execution[event['phase']+':'+event['status']] += 1
    official = reexecute(replay,trace=True,label_sink=sink)
    assert official['engine_sha256'] == cfg['engine_sha256']
    assert official['semantic_sha256'] == row['semantic_replay_sha256']
    assert official['semantic_sha256_v2'] == row['semantic_replay_sha256_v2']
    assert official['final_rewards'] == row['banks'] == row['rewards']
    counts = agent.last['model_counts']
    total = Counter(counts); total.update(counts); total.update(parity)
    assert row['policy'] != 'teacher_task' or counts['task_forwards'] == counts['task_rows'] == 0
    return dict(game=key,passed=True,official=official,live_model_counts=counts,
        primary_model_counts=counts,original_parity_model_counts=dict(parity),
        total_model_counts=dict(total),policy_turns=2*(counts['controller_calls']+719)+(719 if original else 0),
        original_controller_parity_turns=719 if original else 0,parity_kind=row['parity_kind'],
        replay_sha256=row['replay_gzip_sha256'],trace_sha256=row['trace_sha256'],
        actor_actions_reproduced=719,rival_actions_reproduced=719,trace_turns_reproduced=719,
        execution_counts=dict(execution),errors=0)


def _job(out, key):
    out = Path(out).resolve()
    cfg = verify(out)
    plan = read(out/'job-inputs'/(key+'.json'))
    assert plan['parity_kind'] is None, 'No unbudgeted shadow controller'
    assert plan['split'] in ('dev','holdout')
    assert plan['seed'] in cfg['validation_seeds' if plan['split']=='dev' else 'test_seeds']
    bot, rival = out/'agents'/(key+'.py'), out/'sources'/(plan['opponent']+'.py')
    assert plan['game'] == key and sha(bot) == plan['bot_sha256']
    assert sha(rival) == plan['rival_sha256'] == cfg['source_sha256'][rival.name]
    task, digest = task_identity(out,plan,cfg)
    assert digest == plan['task_checkpoint_sha256']
    folder = out/'games'/key
    assert not folder.exists()
    saved_argv, saved_cwd = sys.argv, Path.cwd()
    try:
        sys.argv = [str(ROOT/'run_match.py'),'--bot',str(bot),'--opponent',str(rival),
            '--seed',str(plan['seed']),'--side',str(plan['side']),'--name',key,'--output-root',str(out/'games')]
        os.chdir(ROOT.parent)
        try:
            code = official_entry()
            if code != 0: raise RuntimeError('Actual official entry failed')
        except SystemExit as exc:
            if exc.code not in (0,None):
                raise RuntimeError('Actual official entry failed') from exc
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)
    result, replay = read(folder/'result.json'), read(folder/'replay.json')
    assert result['passed'] and result['done'] and result['statuses'] == ['DONE','DONE']
    assert result['recorded_states'] == 720 and result['action_turns'] == 719 and not result['agent_errors']
    assert result['engine_matches_official_commit'] and result['engine_sha256'] == cfg['engine_sha256']
    assert result['upstream_commit'] == cfg['engine_commit']
    assert result['agent_file_sha256'][str(bot)] == plan['bot_sha256']
    assert result['agent_file_sha256'][str(rival)] == plan['rival_sha256']
    assert result['seed'] == replay['info']['seed'] == plan['seed'] and result['bot_side'] == plan['side']
    assert len(replay['steps']) == 720
    initial = [f['money'] for f in replay['steps'][0][0]['observation']['farms']]
    banks = [f['money'] for f in replay['steps'][-1][0]['observation']['farms']]
    assert banks == result['rewards'] == result['final_money']
    with (folder/'replay.json.gz').open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1) as stream:
            stream.write((folder/'replay.json').read_bytes())
    side = plan['side']; margin = banks[side]-banks[1-side]
    row = dict(plan,passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
        initial_banks=initial,banks=banks,rewards=result['rewards'],bank=banks[side],opponent_bank=banks[1-side],
        initial_bank=initial[side],opponent_initial_bank=initial[1-side],net_profit=banks[side]-initial[side],
        opponent_net_profit=banks[1-side]-initial[1-side],margin=margin,score=1. if margin>0 else .5 if margin==0 else 0.,
        semantic_replay_sha256=result['semantic_replay_sha256'],semantic_replay_sha256_v2=result['semantic_replay_sha256_v2'],
        replay_gzip_sha256=sha(folder/'replay.json.gz'),trace_sha256=sha(out/'traces'/(key+'.jsonl.gz')),
        engine_sha256=cfg['engine_sha256'],crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'])
    write_new(out/'job-results'/(key+'.row.json'),row)
    certificate = audit(out,row,cfg)
    verify(out)
    write_new(out/'job-results'/(key+'.audit.json'),certificate)
    write_new(out/'job-results'/(key+'.complete.json'),dict(game=key,passed=True,
        row_sha256=sha(out/'job-results'/(key+'.row.json')),
        certificate_sha256=sha(out/'job-results'/(key+'.audit.json'))))


def control_difference(out, candidate, baseline):
    """First real request difference on shared pre-divergence official states."""
    for name in ('seed','opponent','side','split'):
        assert candidate[name] == baseline[name]
    assert candidate['policy'] == 'candidate' and baseline['policy'] == 'baseline'
    values = []
    for row in (candidate,baseline):
        path = Path(out)/'games'/row['game']/'replay.json.gz'
        assert sha(path) == row['replay_gzip_sha256']
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            values.append(json.load(stream)['steps'])
    return first_request_difference(*values,candidate['side'])


def first_request_difference(a,b,side):
    assert len(a) == len(b) == 720 and type(side) is int and side in (0,1)
    for step in range(719):
        assert canonical(a[step]) == canonical(b[step]), 'State diverged without prior candidate action difference'
        own = not same_request(a[step+1][side]['action'],b[step+1][side]['action'])
        other = not same_request(a[step+1][1-side]['action'],b[step+1][1-side]['action'])
        assert not other, 'Frozen rival differs on the same state prefix'
        if own:
            return step
    assert canonical(a[-1]) == canonical(b[-1])
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--key',required=True)
    args = parser.parse_args()
    job(args.out,args.key)
