"""Privileged offline checks; reuse the established ordered official replay."""
from collections import Counter
import gzip
import json
from pathlib import Path

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.audit import stable, load
from learning.task_bc_v1.deployment import JointAgent
from decision.audit_market_teacher import observation
from decision.full_audit import reexecute


def audit(out, row, cfg):
    out = Path(out)
    side, arm = row['side'], row['policy']
    assert arm in ('trained', 'teacher_task')
    raw = out / 'games' / row['game'] / 'replay.json.gz'
    assert sha(raw) == row['replay_gzip_sha256']
    assert sha(ROOT / row['agent_relative']) == row['bot_sha256']
    rival_path = out / 'sources' / (row['opponent'] + '.py')
    assert sha(rival_path) == row['rival_sha256'] == cfg['source_sha256'][rival_path.name]
    with gzip.open(raw, 'rt', encoding='utf-8') as stream:
        replay = json.load(stream)
    trace = out / 'traces' / (row['game'] + '.jsonl.gz')
    assert sha(trace) == row['trace_sha256']
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        turns = [json.loads(line) for line in stream]
    assert len(turns) == 719 and len(replay['steps']) == 720
    assert replay['info']['seed'] == row['seed']
    initial = [f['money'] for f in replay['steps'][0][0]['observation']['farms']]
    final = [f['money'] for f in replay['steps'][-1][0]['observation']['farms']]
    assert initial == row['initial_banks'] and final == row['banks']
    assert row['net_profit'] == final[side] - initial[side]
    assert row['opponent_net_profit'] == final[1-side] - initial[1-side]
    model = out / 'model/trained.npz' if arm == 'trained' else None
    assert sha(out / 'model/trained.npz') == cfg['task_checkpoint_sha256']
    assert sha(ROOT / cfg['crop_checkpoint']) == cfg['crop_checkpoint_sha256']
    actor = JointAgent(ROOT / cfg['crop_checkpoint'], cfg['crop_checkpoint_sha256'],
                       model, cfg['task_checkpoint_sha256'] if model else None)
    rival = load(rival_path)
    counts = Counter()
    for i, (before, after) in enumerate(zip(replay['steps'], replay['steps'][1:])):
        assert actor(observation(before, side), replay['configuration']) == after[side]['action']
        assert rival(observation(before, 1-side), replay['configuration']) == after[1-side]['action']
        assert stable(actor.last) == stable(turns[i])
        assert not any(actor.last[k] for k in ('fallbacks', 'load_errors', 'inference_errors'))
        assert actor.last['task_checkpoint_sha256'] == (cfg['task_checkpoint_sha256'] if model else None)
        counts['task_calls'] += len(actor.last['records'])
        counts['crop_calls'] += len(actor.last['crop_records'])
        for rec in actor.last['records']:
            counts['selected_' + rec['op']] += 1
            counts['different_from_conditional_teacher'] += rec['chosen'] != rec['teacher_label']

    def sink(event):
        if event['player'] == side:
            counts[event['phase'] + '_' + event['status']] += 1

    official = reexecute(replay, trace=True, label_sink=sink)
    assert official['engine_sha256'] == cfg['engine_sha256']
    assert official['semantic_sha256'] == row['semantic_replay_sha256']
    assert official['semantic_sha256_v2'] == row['semantic_replay_sha256_v2']
    assert official['final_rewards'] == row['banks'] == row['rewards']
    return dict(game=row['game'], passed=True, official=official, counts=dict(counts),
                actor_actions_reproduced=719, rival_actions_reproduced=719,
                decision_traces_reproduced=719, task_checkpoint_sha256=sha(model) if model else None,
                crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'],
                replay_gzip_sha256=row['replay_gzip_sha256'], trace_sha256=row['trace_sha256'],
                bot_sha256=row['bot_sha256'], rival_sha256=row['rival_sha256'],
                fallback=0, load_errors=0, inference_errors=0)
