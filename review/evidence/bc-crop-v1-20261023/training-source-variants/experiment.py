"""Bounded BC orchestration; reuse round2_worker, Games and official reexecution."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import numpy as np
from .agent import ROOT, TEACHER_SHA, sha
from .features import CROPS, SCHEMA_VERSION

ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
PANEL = {'v1': 'bots/heuristic_v1.py', 'market': 'bots/heuristic_v1_market.py', 'round2': 'models/tuned_round2.py'}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def prepare(out, public):
    out.mkdir(parents=True, exist_ok=False)
    files = subprocess.check_output(['git', '-C', str(public), 'ls-files', '-z']).decode().split('\0')
    protected = {f: sha(public / f) for f in files if f}
    text = '\n'.join((public/f).read_text(encoding='utf-8', errors='ignore') for f in protected if Path(f).suffix in ('.json', '.py', '.md', '.csv'))
    rng = random.Random(2026102307)
    seeds = rng.sample(range(31000000, 32000000), 19)
    assert all(re.search(r'(?<!\d)'+str(s)+r'(?!\d)', text) is None for s in seeds)
    assert sha(ROOT / 'bots/teacher_fertilizer_terminal_v2.py') == TEACHER_SHA
    sources = {'teacher.py': ROOT/'bots/teacher_fertilizer_terminal_v2.py',
               'worker.py': ROOT/'learning/round2_worker.py',
               **{n+'.py': (ROOT/p if (ROOT/p).exists() else public/'kaggriculture'/p) for n,p in PANEL.items()}}
    (out/'sources').mkdir()
    for name, path in sources.items(): shutil.copyfile(path, out/'sources'/name)
    cfg = dict(experiment='limited-crop-bc-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=subprocess.check_output(['git','-C',str(public),'rev-parse','HEAD'], text=True).strip(),
        initial_worktree_clean=not subprocess.check_output(['git','-C',str(public),'status','--porcelain']).strip(),
        engine_commit=ENGINE_COMMIT, engine_sha256=ENGINE_SHA, teacher_sha256=TEACHER_SHA,
        train_seeds=seeds[:8], validation_seeds=seeds[8:10], test_seeds=seeds[10:18], parity_seed=seeds[18],
        seed_novelty='No decimal occurrence in tracked public source/evidence at baseline; all splits disjoint',
        opponents=list(PANEL), seats=[0,1], workers=4,
        training=dict(seed=2026102308, architecture=[289,128,64,5], activation='ReLU', epochs=80,
                      batch_size=256, optimizer='Adam', learning_rate=.001,
                      class_weight='inverse square root train counts, normalized over present classes; absent=0',
                      normalization='train-only mean/std, std floor 1',
                      checkpoint_selection='lowest unweighted validation cross entropy; earliest tie',
                      augmentation='none', extra_demonstration_budget=0),
        smoke=dict(demonstrations='train seed[0] and validation seed[0], market opponent, both seats; included in 60',
                   epochs=5, games=2, policy='smoke-only trained checkpoint, train seed[0], both seats'),
        formal_games=144, demonstration_games=60,
        bootstrap=dict(unit='whole seed, all opponents and seats kept together', draws=20000, seed=2026102309),
        acceptance_A='All 144 games and both file-entry repeats complete 720/719 DONE with no errors; ordered official reexecution; model identity/action reproduction; zero fallbacks; demonstrated actual PLANT and purchase influence',
        acceptance_B='Validation accuracy better than initial and majority baseline; independent trained-minus-initial mean profit 95% whole-seed bootstrap lower bound > 0 and match score mean not lower; teacher gap reported, beating teacher not required',
        scope='Only inner PLANT-task crop selection; teacher opportunity gates, post-choice economic rejection, and all other management retained',
        stop='No final-test-driven changes or additional data; no DAgger/PPO/self-play or market 192 panel',
        source_sha256={n:sha(p) for n,p in sources.items()})
    save(out/'registration.json', cfg)
    save(out/'protection.json', dict(public_files=protected,
        runtime_policies={p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'main.py', *sorted((ROOT/'bots').glob('*.py')), *sorted((ROOT/'models').glob('*.py'))]}))
    print(json.dumps({'prepared': True, 'registration_sha256':sha(out/'registration.json'), 'seeds':seeds}))


def integrity(out):
    cfg = read(out/'registration.json')
    for name,digest in cfg['source_sha256'].items(): assert sha(out/'sources'/name)==digest
    for name,digest in read(out/'protection.json')['runtime_policies'].items(): assert sha(ROOT/name)==digest
    if (out/'collection-source-record.json').exists():
        for name,digest in read(out/'collection-source-record.json')['sources'].items(): assert sha(ROOT/name)==digest
    if (out/'freeze.json').exists():
        for name,digest in read(out/'freeze.json')['runtime_files'].items(): assert sha(ROOT/name)==digest, name
    return cfg


def wrapper(out, key, policy, checkpoint=None, trace=True):
    model = checkpoint.resolve() if checkpoint else None
    tracepath = out/'traces'/(key+'.jsonl.gz')
    assert not tracepath.exists()
    path = out/'agents'/(key+'.py'); path.parent.mkdir(parents=True,exist_ok=True)
    # Relative paths survive a relocated project and an unrelated process CWD.
    code = ('from pathlib import Path\nimport sys\n'
            'ROOT = next(p for p in Path(__file__).resolve().parents if (p / "learning/bc_crop/agent.py").is_file())\n'
            'sys.path.insert(0, str(ROOT))\nfrom learning.bc_crop.agent import CropAgent\n')
    code += '_runner = CropAgent(mode='+repr('teacher' if policy=='teacher' else 'model')
    if model:
        code += ', checkpoint=ROOT / '+repr(model.relative_to(ROOT).as_posix())+', expected_sha='+repr(sha(model))
    if trace:
        code += ', trace=ROOT / '+repr(tracepath.relative_to(ROOT).as_posix())
    code += ')\ndef agent(observation, configuration):\n    return _runner(observation, configuration)\n\nsubmission_agent = agent\n'
    path.write_text(code,encoding='utf-8')
    return path


def games(out, stage):
    cfg = integrity(out)
    sys.path.insert(0, str(ROOT/'learning'))
    from optimize_round2 import Games
    jobspec = []
    if stage=='parity':
        jobspec = [(cfg['parity_seed'],'market',s,p) for s in (0,1) for p in ('original','teacher')]
    elif stage in ('collect-smoke','collect'):
        for seed in cfg['train_seeds']+cfg['validation_seeds']:
            for op in cfg['opponents']:
                for side in (0,1):
                    small=seed in (cfg['train_seeds'][0],cfg['validation_seeds'][0]) and op=='market'
                    if small == (stage=='collect-smoke'): jobspec.append((seed,op,side,'teacher'))
    elif stage in ('smoke','trained-smoke'):
        jobspec=[(cfg['train_seeds'][0],'market',s,'smoke' if stage=='smoke' else p)
                 for s in (0,1) for p in (('trained',) if stage=='smoke' else ('initial','trained'))]
    elif stage=='test':
        assert (out/'freeze.json').is_file()
        jobspec=[(seed,op,s,p) for seed in cfg['test_seeds'] for op in cfg['opponents'] for s in (0,1) for p in ('teacher','initial','trained')]
    else: raise ValueError(stage)
    ledger_path=out/(stage+'-ledger.json')
    assert not ledger_path.exists()
    todo=[]
    for seed,op,side,p in jobspec:
        key=f'{stage}-{seed}-{op}-{side}-{p}'
        model=out/('smoke-model' if p=='smoke' else 'model')/('initial.npz' if p=='initial' else 'trained.npz') if p not in ('original','teacher') else None
        bot=out/'sources/teacher.py' if p=='original' else wrapper(out,key,p,model)
        todo.append((seed,op,side,p,key,bot))
    rows=[]; start=time.perf_counter()
    with Games(out,cfg,stage) as workers,ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        futures={pool.submit(workers.play,bot,p,op,seed,s,out/'games'/key): (key,bot) for seed,op,s,p,key,bot in todo}
        for f in as_completed(futures):
            r=f.result(); key,bot=futures[f]
            r.update(game=key,agent_relative=bot.relative_to(ROOT).as_posix(),
                     trace_sha256=sha(out/'traces'/(key+'.jsonl.gz')) if r['policy']!='original' else None)
            rows.append(r); save(ledger_path,sorted(rows,key=lambda r:r['game']))
            if len(rows)%4==0 or len(rows)==len(todo): print(f'{stage}: {len(rows)}/{len(todo)}',flush=True)
    save(out/(stage+'-timing.json'),dict(wall_seconds=time.perf_counter()-start,games=len(rows)))
    if stage=='parity':
        pairs=[]
        for side in (0,1):
            a,b=[next(r for r in rows if r['side']==side and r['policy']==p) for p in ('original','teacher')]
            assert a['banks']==b['banks'] and a['semantic_replay_sha256_v2']==b['semantic_replay_sha256_v2']
            pairs.append(dict(side=side,semantic_sha256_v2=a['semantic_replay_sha256_v2'],banks=a['banks']))
        save(out/'teacher-parity.json',dict(passed=True,full_games=4,paired_seats=2,pairs=pairs))
    integrity(out)


def dataset(out, smoke=False):
    cfg=integrity(out); assert read(out/'teacher-parity.json')['passed']
    tick=time.perf_counter(); rows=read(out/'collect-smoke-ledger.json')
    if not smoke: rows+=read(out/'collect-ledger.json')
    x=[]; y=[]; seed_ids=[]; game_ids=[]; meta=[]; manifest=[]; outcomes=Counter(); names=None
    from .agent import CropAgent
    from decision.api import observe
    # Schema is obtained from actual decision-time records by the same encoder.
    from .features import encode, schema_hash
    for gi,row in enumerate(sorted(rows,key=lambda r:r['game'])):
        assert row['passed'] and row['trace_sha256']==sha(out/'traces'/(row['game']+'.jsonl.gz'))
        with gzip.open(out/'traces'/(row['game']+'.jsonl.gz'),'rt',encoding='utf-8') as f:
            turns=[json.loads(line) for line in f]
        assert len(turns)==719
        count=0
        for turn in turns:
            for r in turn['records']:
                x.append(r['x']);y.append(CROPS.index(r['teacher']));seed_ids.append(row['seed']);game_ids.append(gi)
                assert r['chosen']==r['teacher']
                meta.append(dict(game=row['game'],seed=row['seed'],step=r['step'],decision=r['decision'],unit=r['unit'],
                                 target=r['target'],label=r['teacher'],outcome=r['outcome']))
                outcomes[r['outcome']]+=1; count+=1
        manifest.append(dict(game=row['game'],seed=row['seed'],opponent=row['opponent'],side=row['side'],
            split='train' if row['seed'] in cfg['train_seeds'] else 'validation',samples=count,
            replay_sha256=row['replay_gzip_sha256'],trace_sha256=row['trace_sha256'],teacher_sha256=TEACHER_SHA))
    # Capture ordered names on one genuine actor observation using the adapter.
    first=rows[0]
    with gzip.open(out/'games'/first['game']/'replay.json.gz','rt',encoding='utf-8') as f: replay=json.load(f)
    from decision.audit_market_teacher import observation
    a=CropAgent(); original_select=a.select
    captured=[]
    def capture(ctx):
        captured.append(encode(ctx)[1]); return original_select(ctx)
    a.controller.__globals__['_bc_select']=capture
    for frame in replay['steps'][:-1]:
        a(observation(frame,first['side']),replay['configuration'])
        if captured: break
    names=captured[0]; assert len(names)==len(x[0])==cfg['training']['architecture'][0]
    dest=out/('smoke-data' if smoke else 'data'); dest.mkdir(exist_ok=False)
    X=np.asarray(x,np.float32);Y=np.asarray(y,np.int64);S=np.asarray(seed_ids,np.int64);G=np.asarray(game_ids,np.int64)
    validation=np.isin(S,cfg['validation_seeds'])
    np.savez_compressed(dest/'dataset.npz',x=X,y=Y,seed=S,game=G,validation=validation)
    with gzip.open(dest/'samples.jsonl.gz','wt',encoding='utf-8') as f:
        for m in meta:f.write(json.dumps(m)+'\n')
    save(dest/'manifest.json',dict(games=manifest,features=names,schema_version=SCHEMA_VERSION,schema_sha256=schema_hash(names),
         dataset_sha256=sha(dest/'dataset.npz'),teacher_sha256=TEACHER_SHA,engine_sha256=ENGINE_SHA))
    counts={split:np.bincount(Y[mask],minlength=5).tolist() for split,mask in [('train',~validation),('validation',validation)]}
    save(dest/'quality.json',dict(samples=len(Y),games=len(rows),class_counts=counts,crops=list(CROPS),
         outcomes=dict(outcomes),all_results_retained=True,whole_seed_split=True,
         training_seeds=sorted(set(S[~validation].tolist())),validation_seeds=sorted(set(S[validation].tolist())),
         missing_train_classes=[CROPS[i] for i,n in enumerate(counts['train']) if not n],
         elapsed_seconds=time.perf_counter()-tick))
    print(json.dumps(read(dest/'quality.json')),flush=True)


def freeze(out):
    cfg=integrity(out)
    assert not (out/'test-ledger.json').exists() and not (out/'freeze.json').exists()
    assert len(read(out/'trained-smoke-ledger.json'))==4
    paths=[*sorted((ROOT/'learning/bc_crop').glob('*.py')),ROOT/'bots/bc_crop_student.py',
           ROOT/'models/bc_crop_v1/checkpoint.json',ROOT/'models/bc_crop_v1/trained.npz',
           ROOT/'decision/api.py',ROOT/'learning/round2_worker.py',
           ROOT/'learning/optimize_round2.py',ROOT/'learning/evaluate_holdout.py',ROOT/'replay_metrics.py',
           ROOT/'decision/full_audit.py',ROOT/'decision/execution.py',ROOT/'decision/semantics.py',ROOT/'decision/verify_mechanisms.py',
           ROOT/'decision/audit_market_teacher.py',ROOT/'decision/verify_market_teacher.py',
           out/'registration.json',out/'data/manifest.json',out/'model/initial.npz',out/'model/trained.npz',
           out/'backend-decision.json',*sorted((out/'sources').glob('*.py'))]
    save(out/'freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        runtime_files={p.relative_to(ROOT).as_posix():sha(p) for p in paths},test_results_absent=True,
        registration_sha256=sha(out/'registration.json'),local_registration_only=True))
    print(json.dumps({'frozen':True,'sha256':sha(out/'freeze.json')}))


def deploy(out):
    dest=ROOT/'models/bc_crop_v1';dest.mkdir(exist_ok=False)
    shutil.copyfile(out/'model/trained.npz',dest/'trained.npz')
    save(dest/'checkpoint.json',dict(file='trained.npz',sha256=sha(dest/'trained.npz'),
        teacher_sha256=TEACHER_SHA,engine_sha256=ENGINE_SHA,architecture=[289,128,64,5],scope='crop choice only'))
    print('Local checkpoint deployed; default main.py unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','parity','collect-smoke','collect','dataset-smoke','dataset','smoke','trained-smoke','deploy','freeze','test'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--public',type=Path)
    args=p.parse_args();out=args.out.resolve()
    if args.mode=='prepare':prepare(out,args.public.resolve())
    elif args.mode in ('dataset','dataset-smoke'):dataset(out,args.mode.endswith('smoke'))
    elif args.mode=='freeze':freeze(out)
    elif args.mode=='deploy':deploy(out)
    else:
        if args.mode in ('collect','collect-smoke'):assert read(out/'teacher-parity.json')['passed']
        games(out,args.mode)
