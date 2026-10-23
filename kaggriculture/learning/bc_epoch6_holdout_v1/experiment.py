"""Freeze one96+2 holdout panel; never train, reselect, retry or change deployment."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.experiment import read, save, integrity, wrapper
from learning.bc_crop.prepare import novelty, git
from learning.bc_crop_diagnostics.panel import bounded_jobs
from learning.bc_crop_economic_diagnostic.experiment import FAMILIES, MODEL_SHA, verify as verify_previous


def admission(cfg, prior_acceptance):
    assert cfg['evaluation_role']=='independent-holdout-fixed-model'
    assert cfg['training_runs']==cfg['parameter_updates']==0
    assert cfg['game_budget']==96 and cfg['entry_budget']==2
    assert cfg['model_sha256']==MODEL_SHA and cfg['model_families']==list(FAMILIES)
    assert cfg['default_change_allowed'] is False and cfg['old_offline_failure_preserved'] is True
    assert prior_acceptance['stage_execution_passed'] and prior_acceptance['exploration_gate_passed']
    assert prior_acceptance['old_offline_gate_passed'] is False and prior_acceptance['old_phase_games']==0
    assert prior_acceptance['checkpoint_sha256']==MODEL_SHA['fixed_epoch6']
    assert prior_acceptance['promotion'] is False
    seeds=cfg['test_seeds']
    assert len(seeds)==8 and len(set(seeds))==8 and all(type(s) is int and 0<=s<2**32 for s in seeds)


def verify(out):
    cfg=integrity(out)
    assert sha(out/'registration.json')==read(out/'freeze.json')['registration_sha256']
    previous=ROOT/cfg['previous_experiment'];verify_previous(previous)
    admission(cfg,read(previous/'acceptance.json'))
    for name,digest in read(out/'protection-before.json')['runtime'].items():
        assert sha(ROOT.parent/name)==digest,name
    for family in FAMILIES:
        child=out/'panels'/family;integrity(child)
        assert sha(child/'model/trained.npz')==MODEL_SHA[family]
        card=read(child/'model/checkpoint.json')
        assert card['file']=='trained.npz' and card['sha256']==MODEL_SHA[family]
    return cfg


def prepare(out,public):
    previous=ROOT/'artifacts/bc-economic-epoch6-v1-20261023'
    old=verify_previous(previous)
    assert not out.exists() and not git(public,'status','--porcelain')
    # Draw once, before looking at any new game. No outcome-dependent redraw.
    seeds=random.Random(2026102325).sample(range(37000000,38000000),8)
    tracked=[n for n in git(public,'ls-files','-z').split('\0') if n]
    scan=novelty(dict(train_seeds=[],validation_seeds=[],test_seeds=seeds,parity_seed=seeds[0]),
                 public,tracked,ROOT/'artifacts')
    cfg=dict(experiment='bc-fixed-epoch6-independent-holdout-v1',
        created_utc=datetime.now(timezone.utc).isoformat(),baseline_head=git(public,'rev-parse','HEAD'),
        previous_experiment=previous.relative_to(ROOT).as_posix(),
        evaluation_role='independent-holdout-fixed-model',new_independent_evidence=True,
        old_offline_failure_preserved=True,prior_diagnostic_exception_acknowledged=True,
        all21_heads_previously_inspected=True,default_change_allowed=False,
        admission_basis='Fixed development-selected epoch6 passed separately registered economic diagnostic. Original offline50percent screen still failed; this new study does not retroactively pass or calibrate it.',
        selection='Same exact epoch6 checkpoint, chosen before all current holdout seeds and games; no alternative heads, models, hyperparameters or strategy changes.',
        model_families=list(FAMILIES),model_sha256=MODEL_SHA,test_seeds=seeds,
        seed_generator='Random(2026102325).sample(range(37000000,38000000),8)',seed_novelty=scan,
        opponents=['v1','market','round2'],seats=[0,1],workers=4,
        game_budget=96,entry_budget=2,training_runs=0,parameter_updates=0,
        entry_tuples=[dict(seed=seeds[0],opponent='market',side=s,model_family='fixed_epoch6') for s in (0,1)],
        engine_sha256=old['engine_sha256'],engine_commit=old['engine_commit'],
        teacher_sha256=old['teacher_sha256'],source_sha256=old['source_sha256'],
        bootstrap=dict(draws=20000,seed=2026102326,unit='whole seed',method='percentile',confidence=.95,quantile_method='linear'),
        primary_outcome='Own final cash minus initial cash3000; pair epoch6-original student at identical seed/opponent/seat.',
        primary_inference='Eight equally weighted complete seed clusters. Within each seed retain all3 opponents and both seats.20k bootstrap samples of8whole seeds with replacement; two-sided95percent percentile CI on mean profit delta only.',
        learned_student_gate='Primary profit CI lower>0 AND average profit delta>=1percent of this holdout original-student mean net profit (baseline must be positive); pooled mean margin/score>=0; market mean profit/margin/score>=0; all execution/privacy/preservation checks pass.',
        scope='If accepted, only a new research improved-student baseline; do not replace default or deployed model, claim teacher superiority, full mechanism learning or large-scale RL readiness.',
        robustness='Disclose all8seed/3opponent/two-model results and48pairs; leave-one-seed-out7seed means descriptive only, never selection or extra pass/fail gate.',
        interpretation='Conditional on fixed development-selected candidate, registered seed generator and3 frozen opponents.8independent clusters, not96independent samples;20k draws add no information. Margin/score/market guards are observed point-estimate nondeclines, not proven population noninferiority.',
        stop='One fixed candidate and one96+2panel; do not peek and adapt. No seed extension, model reselection, retraining or valid losing-game replacement. Infrastructure failure stops dispatch, drains in-flight jobs and preserves evidence; no automatic retry.',
        execution_acceptance='All98games720states/719actions,DONEboth,0Agent errors or model fallbacks; both actors and stable traces reproduced; two official replay passes per game include ordered public/private state, inventory order,v1/v2 and actual economics.2 predetermined filename entries excluded from statistics.')
    admission(cfg,read(previous/'acceptance.json'))
    protected=dict(read(previous/'protection-before.json')['runtime'])
    for base in (previous,ROOT/'learning/bc_crop_economic_diagnostic'):
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:protected[p.relative_to(ROOT.parent).as_posix()]=sha(p)
    out.mkdir(parents=True)
    save(out/'protection-before.json',dict(runtime=protected,public={n:sha(public/n) for n in tracked}))
    shutil.copyfile(previous/'protection.json',out/'protection.json')
    shutil.copytree(previous/'sources',out/'sources');save(out/'registration.json',cfg)
    files=dict(read(previous/'freeze.json')['runtime_files'])
    for p in Path(__file__).parent.glob('*.py'):files[p.relative_to(ROOT).as_posix()]=sha(p)
    for family in FAMILIES:
        child=out/'panels'/family;(child/'model').mkdir(parents=True)
        shutil.copytree(out/'sources',child/'sources');shutil.copyfile(out/'protection.json',child/'protection.json')
        shutil.copyfile(previous/'panels'/family/'model/trained.npz',child/'model/trained.npz')
        save(child/'model/checkpoint.json',dict(file='trained.npz',sha256=MODEL_SHA[family],scope='Fixed model independent holdout; no default replacement'))
        save(child/'registration.json',{**cfg,'model_family':family})
        save(child/'freeze.json',dict(runtime_files=files,registration_sha256=sha(child/'registration.json')))
    for p in out.rglob('*'):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p)
    save(out/'freeze.json',dict(runtime_files=files,registration_sha256=sha(out/'registration.json'),
        new_game_results_absent=True,local_freeze_not_external_timestamp=True))
    verify(out)
    print(f'Registered8 new HOLDOUT seeds {seeds};96+2game cap,0training')


def ready(out):
    cfg=verify(out);pre=read(out/'pre-game-commit.json')
    assert pre['registration_sha256']==sha(out/'registration.json') and pre['freeze_sha256']==sha(out/'freeze.json')
    assert pre['results_absent'] is True and len(pre['commit'])==40
    review=read(out/'pre-review.json')
    assert sha(out/'pre-review.json')==pre['pre_review_sha256'] and review['passed'] is True
    assert review['registration_sha256']==sha(out/'registration.json') and review['freeze_sha256']==sha(out/'freeze.json')
    public=ROOT.parent/'publication/ai-game-lab'
    for name in ('registration.json','freeze.json','pre-review.json'):
        assert subprocess.check_output(['git','-C',str(public),'show',pre['commit']+':review/evidence/'+out.name+'/'+name])==(out/name).read_bytes()
    assert not (out/'failure.json').exists()
    return cfg


def games(out):
    cfg = ready(out)
    assert not (out/'run-started.json').exists()
    assert not (out/'panel-ledger.json').exists()
    assert all(not (out/'panels'/f/'games').exists() for f in FAMILIES)
    save(out/'run-started.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        preregistration_commit=read(out/'pre-game-commit.json')['commit'], maximum_panel_games=96))
    sys.path.insert(0, str(ROOT/'learning'))
    from optimize_round2 import Games
    todo=[]
    for seed in cfg['test_seeds']:
        for op in cfg['opponents']:
            for side in cfg['seats']:
                for family in FAMILIES:
                    child=out/'panels'/family; key=f'holdout-{seed}-{op}-{side}-{family}'
                    todo.append((key,seed,op,side,family,wrapper(child,key,'trained',child/'model/trained.npz')))
    assert len(todo)==96
    rows=[];tick=time.perf_counter()
    with Games(out,cfg,'panel') as pool:
        def execute(job):
            key,seed,op,side,family,bot=job
            return pool.play(bot,'trained',op,seed,side,out/'panels'/family/'games'/key)
        def record(job,row):
            key,seed,op,side,family,bot=job
            row.update(game=key,stage='holdout',model_family=family,checkpoint_sha256=MODEL_SHA[family],
                agent_relative=bot.relative_to(ROOT).as_posix(),
                trace_sha256=sha(out/'panels'/family/'traces'/(key+'.jsonl.gz')))
            rows.append(row)
            save(out/'panel-ledger.json',sorted(rows,key=lambda r:r['game']))
            save(out/'panels'/family/'holdout-ledger.json',sorted([r for r in rows if r['model_family']==family],key=lambda r:r['game']))
            print(f'independent holdout {len(rows)}/96',flush=True)
        errors=bounded_jobs(todo,execute,record,cfg['workers'])
    if errors:
        save(out/'failure.json',dict(stage='panel',errors=errors,completed=len(rows),no_replacement=True))
        raise RuntimeError('Independent holdout stopped; all completed evidence retained')
    save(out/'panel-timing.json',dict(games=len(rows),wall_seconds=time.perf_counter()-tick))
    verify(out)


def entries(out):
    cfg=ready(out);ledger=read(out/'panel-ledger.json')
    assert len(ledger)==96 and not (out/'entry-started.json').exists()
    save(out/'entry-started.json',dict(entry_budget=2,tuples=cfg['entry_tuples']))
    child=out/'panels/fixed_epoch6';bot=ROOT/'bots/bc_crop_student.py';rows=[]
    try:
        for index, target in enumerate(cfg['entry_tuples']):
            matching=[r for r in ledger if all(r[k]==v for k,v in target.items())]
            assert len(matching)==1
            old=matching[0];key=f'entry-test-{index}';trace=child/'traces'/(key+'.jsonl.gz');folder=child/'games'/key
            assert not trace.exists() and not folder.exists()
            cmd=[str(ROOT/'.venv/Scripts/python.exe'),'-B','-X','utf8',str(ROOT/'run_match.py'),
                 '--bot',str(bot),'--opponent',str(child/'sources/market.py'),'--side',str(target['side']),
                 '--seed',str(target['seed']),'--name',key,'--output-root',str(child/'games')]
            done=subprocess.run(cmd,cwd=ROOT.parent,capture_output=True,
                env={**os.environ,'BC_CROP_TRACE':str(trace),'BC_CROP_MODEL_CARD':str(child/'model/checkpoint.json')},
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            (out/(key+'-launcher.log')).write_bytes(done.stdout+done.stderr)
            assert done.returncode==0,'Official filename entry failed'
            result=read(folder/'result.json')
            assert result['passed'] and result['statuses']==['DONE','DONE'] and result['recorded_states']==720
            assert result['final_money']==old['banks'] and result['semantic_replay_sha256_v2']==old['semantic_replay_sha256_v2']
            with gzip.open(folder/'replay.json.gz','wb',compresslevel=1) as f:f.write((folder/'replay.json').read_bytes())
            rows.append({**old,'game':key,'bot_sha256':sha(bot),'agent_relative':bot.relative_to(ROOT).as_posix(),
                'trace_sha256':sha(trace),'replay_gzip_sha256':sha(folder/'replay.json.gz'),
                'entry':'official filename, unrelated cwd','stage':'entry','repeat_of':old['game'],'excluded_from_statistics':True})
            save(child/'entry-test-ledger.json',rows)
            save(out/'entry-ledger.json',rows)
            print(f'official filename entry {index+1}/2: same trajectory',flush=True)
    except Exception as exc:
        save(out/'failure.json',dict(stage='entry',completed=len(rows),exception_type=type(exc).__name__,no_replacement=True))
        raise
    verify(out)


def audit(out):
    from .audit import check
    cfg=ready(out);rows=read(out/'panel-ledger.json')+read(out/'entry-ledger.json')
    assert len(rows)==98 and not (out/'audit.json').exists()
    results=[]
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs=[pool.submit(check,(str(out/'panels'/r['model_family']),r)) for r in rows]
        for future in as_completed(jobs):
            results.append(future.result())
            if len(results)%5==0:print(f'ordered official audit {len(results)}/98',flush=True)
    save(out/'audit.json',dict(passed=True,games=98,unique_game_transitions=98*719,
        actor_and_accounting_reexecutions_per_game=2,rows=sorted(results,key=lambda r:r['game'])))
    verify(out)


def analyze(out):
    from .statistics import summarize
    cfg=ready(out);certificate=read(out/'audit.json')
    assert certificate['passed'] and certificate['games']==98 and not (out/'results.json').exists()
    index={r['game']:r for r in certificate['rows']};assert len(index)==98
    rows=[]
    for r in read(out/'panel-ledger.json'):
        checked=index[r['game']]
        # analysis.check supplies actual accounting aggregates, no raw observation.
        rows.append(checked)
    result=summarize(rows,cfg)
    result.update(registration_sha256=sha(out/'registration.json'),freeze_sha256=sha(out/'freeze.json'),
                  panel_ledger_sha256=sha(out/'panel-ledger.json'),audit_sha256=sha(out/'audit.json'),
                  entry_games=2,training_runs=0,old_offline_gate_passed=False,old_experiment_games=0)
    save(out/'economic-ledger.json',rows);save(out/'results.json',result)
    print('Holdout analysis saved; deployment unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','verify','games','entries','audit','analyze'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--public',type=Path)
    a=p.parse_args();out=a.out.resolve()
    if a.mode=='prepare':prepare(out,a.public.resolve())
    else:globals()[a.mode](out)
