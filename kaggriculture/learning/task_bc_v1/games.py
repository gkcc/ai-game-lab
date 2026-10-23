"""At most12 sequential development games plus2 official filename repeats."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from learning.bc_crop.agent import ROOT,sha
from learning.bc_crop.experiment import read,save
from learning.bc_crop.audit import stable,load
from decision.audit_market_teacher import observation
from decision.full_audit import reexecute
from .deployment import JointAgent
from .experiment import ready


def wrapper(out,key,arm,cfg):
    trace=out/'traces'/(key+'.jsonl.gz');assert not trace.exists()
    model=out/'model'/('initial.npz' if arm=='initial' else 'trained.npz') if arm!='teacher_task' else None
    path=out/'agents'/(key+'.py');path.parent.mkdir(exist_ok=True)
    code=('from pathlib import Path\nimport sys\n'
        'ROOT=next(p for p in Path(__file__).resolve().parents if (p/"learning/task_bc_v1/deployment.py").is_file())\n'
        'sys.path.insert(0,str(ROOT))\nfrom learning.task_bc_v1.deployment import JointAgent\n'
        '_runner=JointAgent(ROOT/'+repr(cfg['crop_checkpoint'])+','+repr(cfg['crop_checkpoint_sha256']))
    if model: code+=',ROOT/'+repr(model.relative_to(ROOT).as_posix())+','+repr(sha(model))
    code+=',trace=ROOT/'+repr(trace.relative_to(ROOT).as_posix())+')\ndef agent(observation,configuration):\n    return _runner(observation,configuration)\n'
    path.write_text(code,encoding='utf-8');return path


def audit(out,row,cfg):
    side=row['side'];arm=row['policy'];raw=out/'games'/row['game']/'replay.json.gz'
    assert sha(raw)==row['replay_gzip_sha256']
    assert sha(ROOT/row['agent_relative'])==row['bot_sha256']
    assert sha(out/'sources/round2.py')==row['rival_sha256']
    with gzip.open(raw,'rt',encoding='utf-8') as stream: replay=json.load(stream)
    trace=out/'traces'/(row['game']+'.jsonl.gz');assert sha(trace)==row['trace_sha256']
    with gzip.open(trace,'rt',encoding='utf-8') as stream: turns=[json.loads(line) for line in stream]
    assert len(turns)==719 and len(replay['steps'])==720
    model=out/'model'/('initial.npz' if arm=='initial' else 'trained.npz') if arm!='teacher_task' else None
    actor=JointAgent(ROOT/cfg['crop_checkpoint'],cfg['crop_checkpoint_sha256'],model,sha(model) if model else None)
    rival=load(out/'sources/round2.py');counts=Counter()
    for i,(before,after) in enumerate(zip(replay['steps'],replay['steps'][1:])):
        assert actor(observation(before,side),replay['configuration'])==after[side]['action']
        assert rival(observation(before,1-side),replay['configuration'])==after[1-side]['action']
        assert stable(actor.last)==stable(turns[i])
        assert not any(actor.last[k] for k in ('fallbacks','load_errors','inference_errors'))
        counts['task_calls']+=len(actor.last['records'])
        counts['crop_calls']+=len(actor.last['crop_records'])
        for rec in actor.last['records']:
            counts['selected_'+rec['op']]+=1
            counts['different_from_conditional_teacher']+=rec['chosen']!=rec['teacher_label']
    def sink(event):
        if event['player']==side:
            counts[event['phase']+'_'+event['status']]+=1
    official=reexecute(replay,trace=True,label_sink=sink)
    assert official['semantic_sha256']==row['semantic_replay_sha256']
    assert official['semantic_sha256_v2']==row['semantic_replay_sha256_v2']
    assert official['final_rewards']==row['banks']
    return dict(game=row['game'],passed=True,official=official,counts=dict(counts),
        actor_actions_reproduced=719,rival_actions_reproduced=719,decision_traces_reproduced=719,
        task_checkpoint_sha256=sha(model) if model else None,crop_checkpoint_sha256=cfg['crop_checkpoint_sha256'],
        fallback=0,load_errors=0,inference_errors=0)


def run(out):
    out=Path(out).resolve();cfg=ready(out,phase='train');gate=read(out/'admission.json')
    assert gate['passed'] and sha(out/'model/initial.npz')==gate['initial_sha256']
    assert sha(out/'model/trained.npz')==gate['trained_sha256']
    assert gate['post_training_review_sha256']==sha(out/'post-training-review.json')
    assert gate['reload_sha256']==sha(out/'reload.json')
    probe=('import importlib.util,hashlib;from pathlib import Path;'
           'root=Path(next(iter(importlib.util.find_spec("kaggle_environments").submodule_search_locations)));'
           'print(hashlib.sha256((root/"envs/kaggriculture/kaggriculture.py").read_bytes()).hexdigest())')
    actual=subprocess.check_output([str(ROOT/'.venv/Scripts/python.exe'),'-B','-c',probe],text=True).strip()
    assert actual==cfg['engine_sha256'],'Installed official rules changed before game dispatch'
    (out/'games-started.json').open('x').write('{}\n')
    sys.path.insert(0,str(ROOT/'learning'))
    from optimize_round2 import Games
    ledger=[];cert=[];attempts=[];tick=time.perf_counter();severe=False
    def attempt(key,seed,side,arm,entry=False):
        attempts.append(dict(game=key,seed=seed,side=side,policy=arm,entry=entry))
        save(out/'attempts.json',attempts)
    with Games(out,cfg,'task-development') as workers:
        for seed in cfg['validation_seeds']:
            for arm in ('trained','teacher_task','initial'):
                for side in (0,1):
                    key=f'dev-{seed}-round2-{side}-{arm}'
                    attempt(key,seed,side,arm)
                    bot=wrapper(out,key,arm,cfg)
                    row=workers.play(bot,arm,'round2',seed,side,out/'games'/key)
                    row.update(game=key,agent_relative=bot.relative_to(ROOT).as_posix(),
                        trace_sha256=sha(out/'traces'/(key+'.jsonl.gz')))
                    ledger.append(row);save(out/'development-ledger.json',ledger)
                    cert.append(audit(out,row,cfg));save(out/'audit.json',cert)
                    print('Development complete '+str(len(ledger))+'/12',flush=True)
            current=[r for r in ledger if r['seed']==seed]
            own=lambda arm:sum(r['bank'] for r in current if r['policy']==arm)/2
            severe=own('trained')<.75*own('teacher_task')
            if severe:break
    # Filename loader repeats are separate from12 statistical rows.
    entries=[]
    if len(ledger)==12:
        for side in (0,1):
            seed=cfg['validation_seeds'][0];key=f'entry-{seed}-{side}-trained';attempt(key,seed,side,'trained',True)
            bot=wrapper(out,key,'trained',cfg)
            cmd=[str(ROOT/'.venv/Scripts/python.exe'),'-B','-X','utf8',str(ROOT/'run_match.py'),
                '--bot',str(bot),'--opponent',str(out/'sources/round2.py'),'--side',str(side),
                '--seed',str(seed),'--name',key,'--output-root',str(out/'games')]
            p=subprocess.run(cmd,cwd=ROOT.parent,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            (out/(key+'-local.log')).write_bytes(p.stdout+p.stderr)
            assert p.returncode==0,'Official file entry failed'
            folder=out/'games'/key;result=read(folder/'result.json')
            old=next(r for r in ledger if r['seed']==seed and r['side']==side and r['policy']=='trained')
            assert result['passed'] and result['statuses']==['DONE','DONE'] and result['recorded_states']==720
            assert result['semantic_replay_sha256_v2']==old['semantic_replay_sha256_v2'] and result['final_money']==old['banks']
            with gzip.open(folder/'replay.json.gz','wb',compresslevel=1) as stream:stream.write((folder/'replay.json').read_bytes())
            row=dict(old,game=key,bot_sha256=sha(bot),agent_relative=bot.relative_to(ROOT).as_posix(),
                trace_sha256=sha(out/'traces'/(key+'.jsonl.gz')),replay_gzip_sha256=sha(folder/'replay.json.gz'))
            entries.append(row);save(out/'entry-ledger.json',entries)
            cert.append(audit(out,row,cfg));save(out/'audit.json',cert)
    summary=dict(completed_development_games=len(ledger),completed_entry_repeats=len(entries),
        severe_regression=severe,independent_holdout=False,wall_seconds=time.perf_counter()-tick,
        official_reexecution_transitions=len(cert)*719,arms={})
    for arm in ('teacher_task','initial','trained'):
        rows=[r for r in ledger if r['policy']==arm]
        summary['arms'][arm]=dict(games=len(rows),mean_net_profit=sum(r['net_profit'] for r in rows)/len(rows),
            mean_margin=sum(r['margin'] for r in rows)/len(rows),
            wins=sum(r['margin']>0 for r in rows),draws=sum(r['margin']==0 for r in rows),losses=sum(r['margin']<0 for r in rows))
    save(out/'game-summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    run(p.parse_args().out)
