"""Both newly generated arms load officially; old observations only, no games."""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import sys

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from learning.bc_crop.audit import stable
from learning.task_bc_v1.deployment import JointAgent
from decision.audit_market_teacher import observation
from .experiment import ready, TASK
from .games import wrapper


def run(out):
    from kaggle_environments.agent import get_last_callable
    out=Path(out).resolve();cfg=ready(out)
    (out/'preflight-started.json').open('x').write('{}\n')
    old_rows=read(TASK/'development-ledger.json');rows=[]
    cwd,paths=Path.cwd(),list(sys.path)
    try:
        os.chdir(ROOT.parent)
        for arm in cfg['arms']:
            old=next(r for r in old_rows if r['seed']==31205138 and r['side']==0 and r['policy']==arm)
            rp=TASK/'games'/old['game']/'replay.json.gz';assert sha(rp)==old['replay_gzip_sha256']
            with gzip.open(rp,'rt',encoding='utf-8') as stream:replay=json.load(stream)
            tp=TASK/'traces'/(old['game']+'.jsonl.gz');assert sha(tp)==old['trace_sha256']
            with gzip.open(tp,'rt',encoding='utf-8') as stream:old_trace=[json.loads(next(stream)),json.loads(next(stream))]
            key='preflight-'+arm;bot=wrapper(out,key,arm,cfg)
            actor=get_last_callable(bot.read_text(encoding='utf-8'),path=str(bot))
            assert '__file__' not in actor.__globals__ and actor is actor.__globals__['submission_agent']
            for i,frame in enumerate(replay['steps'][:2]):
                assert actor(observation(frame,0),deepcopy(replay['configuration']))==replay['steps'][i+1][0]['action']
            trace=out/'traces'/(key+'.jsonl.gz')
            with gzip.open(trace,'rt',encoding='utf-8') as stream:turns=[json.loads(line) for line in stream]
            assert len(turns)==2 and stable(turns)==stable(old_trace)
            # Additional fresh direct instances check deployment input isolation.
            def fresh():
                model=out/'model/trained.npz' if arm=='trained' else None
                return JointAgent(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'],model,
                                  cfg['task_checkpoint_sha256'] if model else None)
            obs=observation(replay['steps'][0],0);conf=replay['configuration']
            altered=deepcopy(obs);altered_cfg=deepcopy(conf);altered_cfg['seed']=987654321
            altered['seed']=123456789;altered['farms'][1]['private']={'shed':{'MELON':999999},'seeds':{'MELON':999999}}
            altered['market']['future_prices']={'MELON':999999}
            base,changed=fresh(),fresh()
            assert base(obs,conf)==changed(altered,altered_cfg)
            assert stable(base.last)==stable(changed.last)
            bad=deepcopy(obs);bad['info']={'seed':3}
            try:fresh()(bad,conf)
            except ValueError:pass
            else:raise AssertionError('Replay info was not rejected')
            rows.append(dict(arm=arm,source_game=old['game'],source_replay_sha256=sha(rp),
                source_trace_sha256=sha(tp),wrapper_relative=bot.relative_to(ROOT).as_posix(),wrapper_sha256=sha(bot),
                trace_sha256=sha(trace),old_action_and_stable_trace_equalities=2,
                private_seed_future_injection_unchanged=True,replay_info_rejected=True))
    finally:
        os.chdir(cwd);sys.path[:]=paths
    result=dict(passed=True,freeze_sha256=sha(out/'freeze.json'),registration_sha256=sha(out/'registration.json'),
        rows=rows,official_loader_calls=2,official_agent_calls=4,direct_agent_valid_calls=4,rejected_info_calls=2,
        official_transitions=0,new_games=0,training_runs=0,non_project_cwd=True,
        boundary='Old consumed development observations only; no new holdout observations or outcomes inspected')
    save(out/'preflight.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
