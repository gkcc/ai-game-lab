"""Offline acceptance: real frozen checkpoints ignore invisible replay fields."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
from .agent import CropAgent, sha
from .audit import stable
from .experiment import read, save, integrity
from decision.audit_market_teacher import observation


def run(out):
    integrity(out)
    row=read(out/'collect-smoke-ledger.json')[0]
    with gzip.open(out/'games'/row['game']/'replay.json.gz','rt',encoding='utf-8') as f:replay=json.load(f)
    results=[]
    for name in ('initial','trained'):
        path=out/'model'/(name+'.npz');agent=CropAgent('model',path,sha(path));cases=0
        for frame in replay['steps'][:-1]:
            actor=observation(frame,row['side']);cfg=deepcopy(replay['configuration'])
            action=agent(actor,cfg);before=deepcopy(agent.last)
            if not before['records']:continue
            changed=deepcopy(frame)
            changed[1-row['side']]['observation']['private']={'seeds':{'MELON':987654321},'shed':{},'inventories':[]}
            cfg['seed']=2147483629
            visible=observation(changed,row['side'])
            visible['farms'][1-row['side']]['private']={'future_answer':'not observable'}
            assert agent(visible,cfg)==action
            assert stable(agent.last)==stable(before)  # real x, logits, choices and reservations
            cases+=1
            if cases==24:break
        assert cases==24
        results.append(dict(model=name,checkpoint_sha256=sha(path),actor_observations=cases,
                            actual_features_logits_choices_and_actions_equal=True))
    save(out/'actual-model-privacy.json',dict(passed=True,source_sha256=sha(Path(__file__)),cases=results,
        verification_only=True,freeze_unchanged=True,test_results_not_used=True))
    print('48 real-checkpoint privacy cases passed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out.resolve())
