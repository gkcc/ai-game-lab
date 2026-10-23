"""Explicit new diagnostic admission, preserving the prior failed offline gate."""
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
from learning.bc_head_fullbatch_v1.experiment import verify as verify_previous

FAMILIES = ('original_student', 'fixed_epoch6')
MODEL_SHA = {
    'original_student': '9806383a2d335fdfe21639f911946b29b36708aba89f7b3961a4f9999b0388fd',
    'fixed_epoch6': 'd497e8284dde1f7b836b1cf88915822f82a96be1f99e7b126d06ea263e281ffd',
}


def admission(cfg, report, acceptance):
    """A new, nonpromotional diagnostic exception is not an old gate pass."""
    assert cfg['diagnostic_exception'] is True and cfg['promotion_allowed'] is False
    assert cfg['training_runs'] == 0 and cfg['game_budget'] == 48 and cfg['entry_budget'] == 2
    assert cfg['model_sha256'] == MODEL_SHA
    assert cfg['model_families'] == list(FAMILIES)
    assert cfg['all21_heads_already_inspected'] is True
    assert report['best_epoch'] == 6 and report['checkpoint_sha256'] == MODEL_SHA['fixed_epoch6']
    assert report['offline_gate_passed'] is False
    assert acceptance['offline_learning_gate_passed'] is False and acceptance['new_games'] == 0
    assert acceptance['actual_candidate_gate_rejected'] is True


def verify(out):
    cfg = integrity(out)
    frozen = read(out/'freeze.json')
    assert sha(out/'registration.json') == frozen['registration_sha256']
    old = ROOT/cfg['previous_experiment']
    verify_previous(old)
    admission(cfg, read(old/'model/report.json'), read(old/'acceptance.json'))
    for name, digest in read(out/'protection-before.json')['runtime'].items():
        assert sha(ROOT.parent/name) == digest, name
    for family in FAMILIES:
        child = out/'panels'/family
        integrity(child)
        assert sha(child/'model/trained.npz') == MODEL_SHA[family]
        card = read(child/'model/checkpoint.json')
        assert card['file'] == 'trained.npz' and card['sha256'] == MODEL_SHA[family]
    return cfg


def prepare(out, public):
    previous = ROOT/'artifacts/bc-head-fullbatch-v1-20261023'
    original = ROOT/'artifacts/bc-crop-v1-20261023-run1'
    old = verify_previous(previous)
    assert not out.exists() and not git(public, 'status', '--porcelain')
    seeds = random.Random(2026102324).sample(range(35000000, 36000000), 4)
    tracked = [n for n in git(public, 'ls-files', '-z').split('\0') if n]
    scan = novelty(dict(train_seeds=seeds, validation_seeds=[], test_seeds=[], parity_seed=seeds[0]),
                   public, tracked, ROOT/'artifacts')
    cfg = dict(experiment='bc-fixed-epoch6-economic-diagnostic-v1',
        created_utc=datetime.now(timezone.utc).isoformat(), baseline_head=git(public, 'rev-parse', 'HEAD'),
        previous_experiment=previous.relative_to(ROOT).as_posix(),
        original_experiment=original.relative_to(ROOT).as_posix(),
        diagnostic_exception=True, promotion_allowed=False, all21_heads_already_inspected=True,
        evaluation_role='new-development-seeds-fixed-model-diagnostic', independent_promotion_evidence=False,
        model_families=list(FAMILIES), model_sha256=MODEL_SHA,
        development_seeds=seeds, seed_generator='Random(2026102324).sample(range(35000000,36000000),4)',
        seed_novelty=scan, opponents=['v1', 'market', 'round2'], seats=[0, 1], workers=4,
        game_budget=48, entry_budget=2, training_runs=0, optimizer_updates=0,
        entry_tuples=[dict(seed=seeds[0], opponent='market', side=s, model_family='fixed_epoch6') for s in (0,1)],
        engine_sha256=old['engine_sha256'], engine_commit=old['engine_commit'],
        teacher_sha256=old['teacher_sha256'], source_sha256=old['source_sha256'],
        selection='Previously selected epoch6 by strict developer validation CE; freeze that exact checkpoint, no new selection among21 inspected heads',
        admission='Explicitly relaxed NEW diagnostic admission only. Old50-percent error screen failed, old experiment remains0games, old gate unchanged. This single model cannot calibrate the screen.',
        metrics=['net_profit','opponent_net_profit','margin','match_score','successful_plants','seed_units_bought','sales_units','sales_revenue','ending_seed_units','ending_warehouse_units'],
        comparison='Fixed epoch6 minus original trained student at identical seed/opponent/seat. Whole seed includes all3 opponents and both seats; four descriptive seed units. Entries excluded.',
        exploration_gate='All4 whole-seed own net-profit deltas>0; pooled margin and match score deltas>=0; market opponent own net-profit/margin/match-score deltas>=0; all execution and preservation checks pass. Never promotion.',
        interpretation='Report all seeds and opponents, including failures. No significance, generalization, threshold calibration, or causal attribution of individual crop changes. Accounting changes include downstream opponent and shared-market effects.',
        stop='Exactly one fixed checkpoint pair; at most48panel+2entrygames; no training, reselection, replacement games, extra seeds, tuning, DAgger/PPO or promotion. Any infrastructure failure stops new dispatch and retains evidence.',
        execution_acceptance='All games720states/719actions,DONE both,0errors/fallbacks; reproduce both actors and traces; full official ordered state and v1/v2 replay hashes;2 predetermined filename entries match trajectories; frozen models/default/history unchanged.')
    admission(cfg, read(previous/'model/report.json'), read(previous/'acceptance.json'))
    protected = dict(read(previous/'protection-before.json')['runtime'])
    for base in (previous, ROOT/'learning/bc_head_fullbatch_v1'):
        for path in base.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                protected[path.relative_to(ROOT.parent).as_posix()] = sha(path)
    out.mkdir(parents=True)
    save(out/'protection-before.json', dict(runtime=protected, public={n:sha(public/n) for n in tracked}))
    shutil.copyfile(previous/'protection.json', out/'protection.json')
    (out/'sources').mkdir()
    for path in (previous/'sources').glob('*.py'): shutil.copyfile(path, out/'sources'/path.name)
    save(out/'registration.json', cfg)
    files = dict(read(previous/'freeze.json')['runtime_files'])
    for p in [*Path(__file__).parent.glob('*.py'), ROOT/'learning/bc_crop_diagnostics/panel.py']:
        files[p.relative_to(ROOT).as_posix()] = sha(p)
    for family, src in [('original_student', original), ('fixed_epoch6', previous)]:
        child = out/'panels'/family
        (child/'model').mkdir(parents=True)
        shutil.copytree(out/'sources', child/'sources')
        shutil.copyfile(out/'protection.json', child/'protection.json')
        shutil.copyfile(src/'model/trained.npz', child/'model/trained.npz')
        save(child/'model/checkpoint.json', dict(file='trained.npz', sha256=MODEL_SHA[family],
            scope='Fixed model economic diagnostic; no promotion'))
        save(child/'registration.json', {**cfg, 'model_family':family})
        save(child/'freeze.json', dict(runtime_files=files, registration_sha256=sha(child/'registration.json')))
    for p in out.rglob('*'):
        if p.is_file(): files[p.relative_to(ROOT).as_posix()] = sha(p)
    save(out/'freeze.json', dict(runtime_files=files, registration_sha256=sha(out/'registration.json'),
        results_absent=True, local_freeze_not_external_timestamp=True))
    verify(out)
    print(f'Registered4 new development seeds {seeds},48+2game cap,0training')


def ready(out):
    cfg = verify(out)
    pre = read(out/'pre-game-commit.json')
    assert pre['registration_sha256'] == sha(out/'registration.json')
    assert pre['freeze_sha256'] == sha(out/'freeze.json')
    assert pre['results_absent'] is True and len(pre['commit']) == 40
    review = read(out/'pre-review.json')
    assert sha(out/'pre-review.json') == pre['pre_review_sha256'] and review['passed'] is True
    assert review['registration_sha256'] == sha(out/'registration.json')
    assert review['freeze_sha256'] == sha(out/'freeze.json')
    # Prove that the named local preregistration commit contains these exact
    # records, rather than accepting a forty-character receipt on trust.
    public=ROOT.parent/'publication/ai-game-lab'
    for name in ('registration.json','freeze.json','pre-review.json'):
        blob=subprocess.check_output(['git','-C',str(public),'show',
            pre['commit']+':review/evidence/'+out.name+'/'+name])
        assert blob==(out/name).read_bytes(),name
    assert not (out/'failure.json').exists()
    return cfg


def games(out):
    cfg = ready(out)
    assert not (out/'run-started.json').exists()
    assert not (out/'panel-ledger.json').exists()
    assert all(not (out/'panels'/f/'games').exists() for f in FAMILIES)
    save(out/'run-started.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        preregistration_commit=read(out/'pre-game-commit.json')['commit'], maximum_panel_games=48))
    sys.path.insert(0, str(ROOT/'learning'))
    from optimize_round2 import Games
    todo=[]
    for seed in cfg['development_seeds']:
        for op in cfg['opponents']:
            for side in cfg['seats']:
                for family in FAMILIES:
                    child=out/'panels'/family; key=f'dev-{seed}-{op}-{side}-{family}'
                    todo.append((key,seed,op,side,family,wrapper(child,key,'trained',child/'model/trained.npz')))
    assert len(todo)==48
    rows=[];tick=time.perf_counter()
    with Games(out,cfg,'panel') as pool:
        def execute(job):
            key,seed,op,side,family,bot=job
            return pool.play(bot,'trained',op,seed,side,out/'panels'/family/'games'/key)
        def record(job,row):
            key,seed,op,side,family,bot=job
            row.update(game=key,stage='development',model_family=family,checkpoint_sha256=MODEL_SHA[family],
                agent_relative=bot.relative_to(ROOT).as_posix(),
                trace_sha256=sha(out/'panels'/family/'traces'/(key+'.jsonl.gz')))
            rows.append(row)
            save(out/'panel-ledger.json',sorted(rows,key=lambda r:r['game']))
            save(out/'panels'/family/'dev-ledger.json',sorted([r for r in rows if r['model_family']==family],key=lambda r:r['game']))
            print(f'fixed-model diagnostic {len(rows)}/48',flush=True)
        errors=bounded_jobs(todo,execute,record,cfg['workers'])
    if errors:
        save(out/'failure.json',dict(stage='panel',errors=errors,completed=len(rows),no_replacement=True))
        raise RuntimeError('Diagnostic stopped; all completed evidence retained')
    save(out/'panel-timing.json',dict(games=len(rows),wall_seconds=time.perf_counter()-tick))
    verify(out)


def entries(out):
    cfg=ready(out);ledger=read(out/'panel-ledger.json')
    assert len(ledger)==48 and not (out/'entry-started.json').exists()
    save(out/'entry-started.json',dict(entry_budget=2,tuples=cfg['entry_tuples']))
    child=out/'panels/fixed_epoch6';bot=ROOT/'bots/bc_crop_student.py';rows=[]
    try:
        for index, target in enumerate(cfg['entry_tuples']):
            matching=[r for r in ledger if all(r[k]==v for k,v in target.items())]
            assert len(matching)==1
            old=matching[0];key=f'entry-dev-{index}';trace=child/'traces'/(key+'.jsonl.gz');folder=child/'games'/key
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
            save(child/'entry-dev-ledger.json',rows)
            save(out/'entry-ledger.json',rows)
            print(f'official filename entry {index+1}/2: same trajectory',flush=True)
    except Exception as exc:
        save(out/'failure.json',dict(stage='entry',completed=len(rows),exception_type=type(exc).__name__,no_replacement=True))
        raise
    verify(out)


def audit(out):
    from .analysis import check
    cfg=ready(out);rows=read(out/'panel-ledger.json')+read(out/'entry-ledger.json')
    assert len(rows)==50 and not (out/'audit.json').exists()
    results=[]
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs=[pool.submit(check,(str(out/'panels'/r['model_family']),r)) for r in rows]
        for future in as_completed(jobs):
            results.append(future.result())
            if len(results)%5==0:print(f'ordered official audit {len(results)}/50',flush=True)
    save(out/'audit.json',dict(passed=True,games=50,unique_game_transitions=50*719,
        actor_and_accounting_reexecutions_per_game=2,rows=sorted(results,key=lambda r:r['game'])))
    verify(out)


def analyze(out):
    from .analysis import summarize
    cfg=ready(out);certificate=read(out/'audit.json')
    assert certificate['passed'] and certificate['games']==50 and not (out/'results.json').exists()
    index={r['game']:r for r in certificate['rows']};assert len(index)==50
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
    print('Economic analysis saved; no promotion permitted')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','verify','games','entries','audit','analyze'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--public',type=Path)
    a=p.parse_args();out=a.out.resolve()
    if a.mode=='prepare':prepare(out,a.public.resolve())
    else:globals()[a.mode](out)
