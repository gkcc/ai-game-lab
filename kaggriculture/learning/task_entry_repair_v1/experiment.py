"""Separate historical filename repair; immutable task training remains failed."""
import argparse
from datetime import datetime, timezone
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save
from learning.task_bc_v1.experiment import verify as verify_previous

PREVIOUS = ROOT/'artifacts/task-bc-v1-20261023-run1'
PUBLIC = ROOT.parent/'publication/ai-game-lab'


def git(*args):
    return subprocess.check_output(['git','-C',str(PUBLIC),*args])


def installed_files():
    package=Path(next(iter(importlib.util.find_spec('kaggle_environments').submodule_search_locations)))
    return {'rules':package/'envs/kaggriculture/kaggriculture.py','loader':package/'agent.py'}


def prepare(out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve all previous attempts')
    assert not git('status','--porcelain').strip()
    old_cfg=verify_previous(PREVIOUS,history=True)
    old_outcome=read(PREVIOUS/'stage-outcome.json')
    assert old_outcome['stage_acceptance_passed'] is False
    advice=read(PREVIOUS/'next-stage-advice.json')
    assert advice['budgets']['maximum_complete_game_attempts']==2
    official=installed_files()
    assert sha(official['rules'])==old_cfg['engine_sha256']
    assert sha(official['loader'])==read(PREVIOUS/'independent-entry-defect.json')['evidence_sha256']['installed_official_loader']
    protected=read(PREVIOUS/'protection-before.json')['runtime'].copy()
    for base in (PREVIOUS,ROOT/'learning/task_bc_v1'):
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:protected[p.relative_to(ROOT).as_posix()]=sha(p)
    protected['TASK_BC_V1.zh-CN.md']=sha(ROOT/'TASK_BC_V1.zh-CN.md')
    files=read(PREVIOUS/'freeze.json')['runtime_files'].copy()
    for p in (ROOT/'learning/task_entry_repair_v1').iterdir():
        if p.is_file() and p.suffix in ('.py','.md'):files[p.relative_to(ROOT).as_posix()]=sha(p)
    cfg=dict(experiment='task-entry-repair-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git('rev-parse','HEAD').decode().strip(),initial_worktree_clean=True,
        previous_root=PREVIOUS.relative_to(ROOT).as_posix(),prior_stage_acceptance_passed=False,
        engine_commit=old_cfg['engine_commit'],engine_sha256=old_cfg['engine_sha256'],
        official_loader_sha256=sha(official['loader']),
        task_checkpoint_sha256=advice['model_sha256'],crop_checkpoint=old_cfg['crop_checkpoint'],
        crop_checkpoint_sha256=old_cfg['crop_checkpoint_sha256'],seed=31205138,seats=[0,1],
        maximum_game_attempts=2,per_game_timeout_seconds=900,training_runs=0,new_seeds=0,new_evaluation_panels=0,
        interpretation='Historical official filename reproductions, not new economic samples or repaired old acceptance',
        stop='Any regression, game, identity or audit failure stops dispatch; no retry or replacement',
        advice_sha256=sha(PREVIOUS/'next-stage-advice.json'),
        acceptance=['Real old official loader failure, new official/Worker/direct two-observation action and trace agreement',
                    'Two full official filename games;720states/719actions/DONE both;zero errors/fallback',
                    'All actions, semantic v1/v2, final banks/rewards and stable traces equal old trained games',
                    'Official ordered replay of both games; independent two saved replay audits separately',
                    'Frozen strategies, models, default and every protected old file unchanged'])
    out.mkdir(parents=True)
    for sub in ('sources','model','agents','traces','games'): (out/sub).mkdir()
    for rel in ('sources/round2.py','sources/worker.py','model/trained.npz'):
        shutil.copyfile(PREVIOUS/rel,out/rel)
    assert sha(out/'model/trained.npz')==cfg['task_checkpoint_sha256']
    cfg['copied_files']={rel:sha(out/rel) for rel in ('sources/round2.py','sources/worker.py','model/trained.npz')}
    save(out/'registration.json',cfg)
    public={rel:sha(PUBLIC/rel) for rel in git('ls-files','-z').decode().split('\0') if rel}
    save(out/'protection-before.json',dict(runtime=protected,public=public))
    save(out/'freeze.json',dict(runtime_files=files,registration_sha256=sha(out/'registration.json'),
        protection_sha256=sha(out/'protection-before.json'),local_freeze_not_external_timestamp=True,
        execution_results_absent=True))
    return verify(out,history=True)


def verify(out,history=False):
    out=Path(out);cfg=read(out/'registration.json');freeze=read(out/'freeze.json')
    assert sha(out/'registration.json')==freeze['registration_sha256']
    assert sha(out/'protection-before.json')==freeze['protection_sha256']
    for rel,digest in freeze['runtime_files'].items():assert sha(ROOT/rel)==digest,rel
    for rel,digest in cfg['copied_files'].items():assert sha(out/rel)==digest,rel
    assert sha(ROOT/cfg['crop_checkpoint'])==cfg['crop_checkpoint_sha256']
    actual=installed_files()
    assert sha(actual['rules'])==cfg['engine_sha256']
    assert sha(actual['loader'])==cfg['official_loader_sha256']
    if history:
        for rel,digest in read(out/'protection-before.json')['runtime'].items():assert sha(ROOT/rel)==digest,rel
    return cfg


def ready(out):
    out=Path(out);cfg=verify(out)
    review,receipt=read(out/'pre-review.json'),read(out/'pre-execution-commit.json')
    assert review['passed'] and review['freeze_sha256']==sha(out/'freeze.json')
    for name in ('registration.json','freeze.json','pre-review.json'):
        assert git('show',receipt['commit']+':review/evidence/'+out.name+'/'+name)==(out/name).read_bytes()
    return cfg


def run(out):
    from learning.task_bc_v1.games import audit
    from learning.bc_crop.audit import stable
    from .entry import generate_wrapper
    out=Path(out).resolve();cfg=ready(out);verify(out,history=True)
    assert read(out/'regressions.json')['passed']
    assert read(out/'regressions.json')['freeze_sha256']==sha(out/'freeze.json')
    (out/'games-started.json').open('x').write('{}\n')
    previous=ROOT/cfg['previous_root'];old_rows=read(previous/'development-ledger.json')
    ledger=[];certificates=[];attempts=[]
    def trace(path):
        with gzip.open(path,'rt',encoding='utf-8') as stream:return [stable(json.loads(line)) for line in stream]
    for side in cfg['seats']:
        verify(out)
        key=f"repair-{cfg['seed']}-{side}-trained"
        attempts.append(dict(game=key,seed=cfg['seed'],side=side,purpose='historical_filename_reproduction'))
        assert len(attempts)<=cfg['maximum_game_attempts']
        save(out/'attempts.json',attempts)
        try:
            bot=generate_wrapper(out,key,cfg)
            cmd=[str(ROOT/'.venv/Scripts/python.exe'),'-B','-X','utf8',str(ROOT/'run_match.py'),
                '--bot',str(bot),'--opponent',str(out/'sources/round2.py'),'--seed',str(cfg['seed']),
                '--side',str(side),'--name',key,'--output-root',str(out/'games')]
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            proc=subprocess.run(cmd,cwd=ROOT.parent,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                timeout=cfg['per_game_timeout_seconds'],creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            (out/(key+'-local.log')).write_bytes(proc.stdout+proc.stderr)
            assert proc.returncode==0,'Official filename run failed'
            folder=out/'games'/key;r=read(folder/'result.json')
            old=next(x for x in old_rows if x['seed']==cfg['seed'] and x['side']==side and x['policy']=='trained')
            assert r['passed'] and r['done'] and r['statuses']==['DONE','DONE']
            assert r['recorded_states']==720 and r['action_turns']==719 and not r['agent_errors']
            assert r['engine_sha256']==cfg['engine_sha256'] and r['upstream_commit']==cfg['engine_commit']
            assert r['agent_file_sha256'][str(bot)]==sha(bot)
            assert r['agent_file_sha256'][str(out/'sources/round2.py')]==sha(out/'sources/round2.py')
            for name in ('semantic_replay_sha256','semantic_replay_sha256_v2'):assert r[name]==old[name],name
            assert r['final_money']==r['rewards']==old['banks']
            replay=read(folder/'replay.json')
            with gzip.open(previous/'games'/old['game']/'replay.json.gz','rt',encoding='utf-8') as stream:old_replay=json.load(stream)
            assert [[s['action'] for s in t] for t in replay['steps'][1:]]==[[s['action'] for s in t] for t in old_replay['steps'][1:]]
            assert trace(out/'traces'/(key+'.jsonl.gz'))==trace(previous/'traces'/(old['game']+'.jsonl.gz'))
            with gzip.open(folder/'replay.json.gz','wb',compresslevel=1) as stream:stream.write((folder/'replay.json').read_bytes())
            row=dict(old,game=key,bot_sha256=sha(bot),agent_relative=bot.relative_to(ROOT).as_posix(),
                trace_sha256=sha(out/'traces'/(key+'.jsonl.gz')),replay_gzip_sha256=sha(folder/'replay.json.gz'),
                reference_game=old['game'],historical_reproduction=True)
            certificate=audit(out,row,cfg)
            ledger.append(row);certificates.append(certificate)
            save(out/'entry-ledger.json',ledger);save(out/'audit.json',certificates)
            print('Historical filename reproduction complete '+str(len(ledger))+'/2',flush=True)
        except Exception as exc:
            save(out/'failure.json',dict(passed=False,game=key,error_type=type(exc).__name__,attempts=len(attempts),
                 complete_entries=len(ledger),dispatch_stopped=True,prior_stage_acceptance_passed=False))
            raise
    verify(out,history=True)
    summary=dict(passed=True,completed_historical_filename_reproductions=len(ledger),attempts=len(attempts),
        root_official_reexecution_transitions=len(certificates)*719,training_runs=0,new_economic_samples=0,
        prior_stage_acceptance_passed=False,default_changed=False,old_development_mean_increment=-155,
        ledger_sha256=sha(out/'entry-ledger.json'),audit_sha256=sha(out/'audit.json'),
        regressions_sha256=sha(out/'regressions.json'),freeze_sha256=sha(out/'freeze.json'))
    save(out/'result.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','verify','run'))
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare(a.out)
    elif a.mode=='verify':verify(a.out,history=True)
    else:run(a.out)
