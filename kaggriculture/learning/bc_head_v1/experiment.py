"""Register one head fit and at most12 developer games plus2 entry repeats."""
import argparse
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
from learning.bc_crop.agent import ROOT,sha
from learning.bc_crop.experiment import read,save,integrity,wrapper
from learning.bc_crop_diagnostics.panel import bounded_jobs,verify as verify_diagnostics


def verify(out):
    cfg=integrity(out)
    assert sha(out/'registration.json')==read(out/'freeze.json')['registration_sha256']
    integrity(ROOT/cfg['original_experiment'])
    previous=verify_diagnostics(ROOT/cfg['diagnostics_experiment'])
    assert previous['checkpoint_sha256']==sha(ROOT/cfg['original_experiment']/'model/trained.npz')
    assert previous['engine_sha256']==cfg['engine_sha256'] and previous['source_sha256']==cfg['source_sha256']
    for name,digest in cfg['input_sha256'].items():assert sha(ROOT/name)==digest,name
    return cfg


def prepare(out,public):
    source=ROOT/'artifacts/bc-crop-v1-20261023-run1';diagnostics=ROOT/'artifacts/bc-diagnostics-v1-20261023'
    original=integrity(source);assert read(diagnostics/'acceptance.json')['passed'];verify_diagnostics(diagnostics)
    assert not out.exists()
    cfg=dict(original)
    cfg.update(experiment='bc-output-head-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=subprocess.check_output(['git','-C',str(public),'rev-parse','HEAD'],text=True).strip(),
        evaluation_role='development-reuse',new_independent_evidence=False,
        original_experiment=source.relative_to(ROOT).as_posix(),diagnostics_experiment=diagnostics.relative_to(ROOT).as_posix(),
        training=dict(seed=2026102321,epochs=20,batch_size=256,learning_rate=.0003,optimizer='Adam',
            architecture=[289,128,64,5],update_only=['w2','b2'],frozen=['w0','b0','w1','b1','mean','scale'],
            class_weights='original inverse square root train counts, normalized over present classes',
            sampling='all original train calls once per epoch in seeded permutation; no deduplication/reweighting/augmentation',
            precision='highest FP32 GPU forward/gradients; selection uses deployment float64 CPU logits',
            selection='minimum unweighted original validation CE over epoch0 and epochs1..20; strict improvement, earliest tie'),
        development_seeds=read(diagnostics/'panel-registration.json')['development_seeds'],
        training_budget=1,optimizer_update_budget=720,new_game_budget=14,development_games=12,file_entry_repeats=2,
        formal_games=0,demonstration_games=0,
        offline_gate='selected epoch>0; errors on original160 eligible train strawberry errors <=80; developer val accuracy drop<=.01; val exact-unseen accuracy drop<=.02',
        development_gate='candidate minus historical frozen student: net profit positive in each whole seed; pooled margin/score>=0; market mean profit/margin/score>=0. Descriptive only, no statistical significance or promotion.',
        baseline_reuse='Exactly12 student rows from previous36-game developer panel; no new baseline games; hashes frozen here',
        acceptance_A='Only w2/b2 can change; baseline weights/data/controller/default preserved; finite nonzero GPU gradients and actual parameter updates; exact identity reload; all permitted games720/719 DONE, zero fallback, complete official ordered reexecution; two real file-entry repeats match developer games.',
        acceptance_B='Development gates only. Does not establish new independent improvement.',
        stop='One fit, no sweep/retry of training hyperparameters; epoch0 or offline failure stops before games. At most14 new games. No formal holdout, DAgger/PPO, data additions or deployment promotion.',
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [source/'data/dataset.npz',source/'data/manifest.json',source/'data/samples.jsonl.gz',source/'model/trained.npz',diagnostics/'panel-ledger.json',diagnostics/'panel-audit.json',diagnostics/'panel-results.json',diagnostics/'panel-registration.json',diagnostics/'panel-freeze-v2.json',diagnostics/'acceptance.json']})
    # Remove obsolete inherited smoke/bootstrap thresholds: this is a new and
    # explicitly non-independent development fit, not a replica of the old fit.
    for key in ('smoke','bootstrap','parity_seed','seed_novelty'):cfg.pop(key,None)
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    for p in (source/'sources').glob('*.py'):shutil.copyfile(p,out/'sources'/p.name)
    shutil.copyfile(source/'protection.json',out/'protection.json')
    save(out/'registration.json',cfg)
    files=dict(read(source/'freeze.json')['runtime_files'])
    for p in list((ROOT/'learning/bc_head_v1').glob('*.py'))+[ROOT/'learning/bc_crop_diagnostics/panel.py',out/'registration.json',out/'protection.json']:
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    save(out/'freeze.json',dict(runtime_files=files,registration_sha256=sha(out/'registration.json'),
        training_and_game_results_absent=True,local_freeze_not_external_timestamp=True))
    print(json.dumps({'prepared':True,'training_runs':1,'max_new_games':14,'formal_games':0}))


def model_gate(out):
    cfg=verify(out);report=read(out/'model/report.json')
    assert report['offline_gate_passed'] and report['best_epoch']>0
    assert sha(out/'model/trained.npz')==report['checkpoint_sha256']
    reload=read(out/'reload-verification.json')
    assert reload['passed'] and reload['checkpoint_sha256']==report['checkpoint_sha256']
    return cfg


def freeze_candidate(out):
    model_gate(out)
    assert not (out/'candidate-freeze.json').exists() and not (out/'games').exists()
    save(out/'candidate-freeze.json',dict(new_game_results_absent=True,
        registration_sha256=sha(out/'registration.json'),source_freeze_sha256=sha(out/'freeze.json'),
        files={n:sha(out/n) for n in ['model/trained.npz','model/report.json','model/checkpoint.json','reload-verification.json']}))


def fit_gate(out):
    cfg=model_gate(out);frozen=read(out/'candidate-freeze.json')
    assert frozen['registration_sha256']==sha(out/'registration.json') and frozen['source_freeze_sha256']==sha(out/'freeze.json')
    for name,digest in frozen['files'].items():assert sha(out/name)==digest,name
    return cfg


def games(out):
    cfg=fit_gate(out);assert not (out/'dev-ledger.json').exists() and not (out/'games').exists()
    sys.path.insert(0,str(ROOT/'learning'));from optimize_round2 import Games
    todo=[]
    for seed in cfg['development_seeds']:
        for op in cfg['opponents']:
            for side in cfg['seats']:
                key=f'dev-{seed}-{op}-{side}-trained'
                todo.append((key,seed,op,side,wrapper(out,key,'trained',out/'model/trained.npz')))
    assert len(todo)==cfg['development_games']==12
    rows=[]
    with Games(out,cfg,'dev') as workers:
        def execute(job):
            key,seed,op,side,bot=job;return workers.play(bot,'trained',op,seed,side,out/'games'/key)
        def record(job,r):
            key,seed,op,side,bot=job;r.update(game=key,agent_relative=bot.relative_to(ROOT).as_posix(),trace_sha256=sha(out/'traces'/(key+'.jsonl.gz')))
            rows.append(r);save(out/'dev-ledger.json',sorted(rows,key=lambda r:r['game']))
            print(f'developer candidate {len(rows)}/12',flush=True)
        errors=bounded_jobs(todo,execute,record,cfg['workers'])
    if errors:
        save(out/'dev-failure.json',dict(errors=errors,completed=len(rows)));raise RuntimeError('Developer panel failed; no refill/replacement')
    verify(out)


def entries(out):
    cfg=fit_gate(out);ledger=read(out/'dev-ledger.json');assert len(ledger)==12
    assert not (out/'entry-dev-ledger.json').exists()
    bot=ROOT/'bots/bc_crop_student.py';rows=[]
    for side in (0,1):
        old=next(r for r in ledger if r['seed']==cfg['development_seeds'][0] and r['opponent']=='market' and r['side']==side)
        key=f'entry-dev-{side}';trace=out/'traces'/(key+'.jsonl.gz');folder=out/'games'/key
        assert not trace.exists() and not folder.exists()
        cmd=[str(ROOT/'.venv/Scripts/python.exe'),'-B','-X','utf8',str(ROOT/'run_match.py'),'--bot',str(bot),
             '--opponent',str(out/'sources/market.py'),'--side',str(side),'--seed',str(old['seed']),'--name',key,'--output-root',str(out/'games')]
        done=subprocess.run(cmd,cwd=ROOT.parent,capture_output=True,
            env={**os.environ,'BC_CROP_TRACE':str(trace),'BC_CROP_MODEL_CARD':str(out/'model/checkpoint.json')},
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        (out/(key+'-launcher.log')).write_bytes(done.stdout+done.stderr)
        if done.returncode:raise RuntimeError('File entry failed; retained local launcher log')
        result=read(folder/'result.json')
        assert result['passed'] and result['statuses']==['DONE','DONE'] and result['recorded_states']==720
        assert result['final_money']==old['banks'] and result['semantic_replay_sha256_v2']==old['semantic_replay_sha256_v2']
        with gzip.open(folder/'replay.json.gz','wb',compresslevel=1) as f:f.write((folder/'replay.json').read_bytes())
        rows.append({**old,'game':key,'bot_sha256':sha(bot),'agent_relative':bot.relative_to(ROOT).as_posix(),
                     'trace_sha256':sha(trace),'replay_gzip_sha256':sha(folder/'replay.json.gz'),'entry':'official filename, unrelated cwd'})
        save(out/'entry-dev-ledger.json',rows)
        print(f'official file entry {side}: same trajectory',flush=True)
    verify(out)


def analyze(out):
    cfg=fit_gate(out);rows=read(out/'dev-ledger.json');audit=read(out/'audit-dev-entry-dev.json')
    assert len(rows)==12 and audit['passed'] and audit['games']==14 and not (out/'results.json').exists()
    baseline=[r for r in read(ROOT/cfg['diagnostics_experiment']/'panel-ledger.json') if r['policy']=='student']
    def key(r):return r['seed'],r['opponent'],r['side']
    indexed={key(r):r for r in baseline};assert len(indexed)==12 and set(indexed)=={key(r) for r in rows}
    def score(r):return 1. if r['margin']>0 else .5 if r['margin']==0 else 0.
    paired=[dict(seed=r['seed'],opponent=r['opponent'],side=r['side'],net_profit=r['net_profit']-indexed[key(r)]['net_profit'],
                 margin=r['margin']-indexed[key(r)]['margin'],match_score=score(r)-score(indexed[key(r)])) for r in rows]
    def avg(part):return {k:mean(r[k] for r in part) for k in ('net_profit','margin','match_score')}
    seeds={str(s):avg([r for r in paired if r['seed']==s]) for s in cfg['development_seeds']}
    opponents={o:avg([r for r in paired if r['opponent']==o]) for o in cfg['opponents']};overall=avg(paired)
    gate=all(r['net_profit']>0 for r in seeds.values()) and all(overall[k]>=0 for k in ('margin','match_score')) and all(v>=0 for v in opponents['market'].values())
    save(out/'results.json',dict(development_only=True,new_independent_evidence=False,development_gate_passed=gate,
        new_games=14,reused_baseline_games=12,file_repeats_excluded_from_statistics=True,paired=paired,by_seed=seeds,by_opponent=opponents,
        candidate_minus_student=overall,checkpoint_sha256=sha(out/'model/trained.npz'),audit_sha256=sha(out/'audit-dev-entry-dev.json')))
    print(json.dumps({'development_gate_passed':gate,'candidate_minus_student':overall}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','freeze-candidate','games','entries','analyze'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--public',type=Path)
    a=p.parse_args();out=a.out.resolve()
    if a.mode=='prepare':prepare(out,a.public.resolve())
    else:{'freeze-candidate':freeze_candidate,'games':games,'entries':entries,'analyze':analyze}[a.mode](out)
