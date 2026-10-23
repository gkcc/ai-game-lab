"""One bounded development diagnostic: fixed weights, 2 seeds, 36 games.

Reuses the official worker and ordered reexecutor. Never retrains or promotes a
policy. Registration and source freeze precede all new development games.
"""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import random
import shutil
from statistics import mean
import subprocess
import sys
import time

from learning.bc_crop.agent import ROOT, CropAgent, sha
from learning.bc_crop.experiment import ENGINE_SHA, ENGINE_COMMIT, PANEL, read, save, integrity
from learning.bc_crop.prepare import novelty, git
from learning.bc_crop.audit import load, stable
from .probe import StrawberryCorrection


def verify(out):
    cfg = read(out/'panel-registration.json')
    frozen = read(out/'panel-freeze.json')
    assert sha(out/'panel-registration.json') == frozen['registration_sha256']
    for name, digest in frozen['runtime_files'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in cfg['source_sha256'].items():
        assert sha(out/'sources'/name) == digest, name
    for name, digest in frozen['analysis_inputs'].items():
        assert sha(out/name) == digest, name
    integrity(ROOT/cfg['original_experiment'])
    return cfg


def prepare(out, source, public):
    assert not (out/'panel-registration.json').exists()
    old = integrity(source)
    diagnosis = read(out/'existing-diagnostics.json')
    assert diagnosis['passed']
    assert diagnosis['selected_probe_teacher_crop']=='STRAWBERRY'
    # Fixed generator, drawn once before any new match. Two DEVELOPMENT seeds.
    seeds = random.Random(2026102319).sample(range(33000000, 34000000), 2)
    tracked = [p for p in git(public, 'ls-files', '-z').split('\0') if p]
    # novelty's inputs are just flattened by that helper; this is NOT a split
    # registration and bypasses none of prepare.seed_config's holdout validation.
    scan = novelty(dict(train_seeds=seeds, validation_seeds=[], test_seeds=[], parity_seed=seeds[0]),
                   public, tracked, ROOT/'artifacts')
    (out/'sources').mkdir(exist_ok=False)
    for p in (source/'sources').glob('*.py'):
        shutil.copyfile(p, out/'sources'/p.name)
    model = source/'model/trained.npz'
    cfg = dict(experiment='bc-crop-strawberry-diagnostic-v1',
               created_utc=datetime.now(timezone.utc).isoformat(), baseline_head=git(public,'rev-parse','HEAD'),
               evaluation_role='development-counterfactual', new_independent_evidence=False,
               original_experiment=source.relative_to(ROOT).as_posix(),
               model_relative=model.relative_to(ROOT).as_posix(), checkpoint_sha256=sha(model),
               engine_sha256=ENGINE_SHA, engine_commit=ENGINE_COMMIT,
               teacher_sha256=old['teacher_sha256'], source_sha256=old['source_sha256'],
               development_seeds=seeds, seed_generator='Random(2026102319).sample(range(33000000,34000000),2)',
               seed_novelty=scan, opponents=list(PANEL), seats=[0,1],
               policies=['teacher','student','probe'], workers=4, game_budget=36, training_runs=0,
               intervention='On the current student-reached decision context, override only when conditional frozen teacher selects STRAWBERRY and frozen model disagrees. Same controller, no model training. Diagnostic oracle, not a student or fallback.',
               selection='Prewritten train-only mean per-seed teacher-score-gap ranking; selected STRAWBERRY. Teacher preference gap is not official cash regret.',
               qualification='160 eligible train errors, 23 exact vectors; 128 calls from seven vectors recurring across all eight seeds. Repetition is not eight distinct states.',
               metrics=['net_profit','opponent_bank','margin','match_score','correction_calls','successful_plants','seed_units_bought','ending_seed_units','sales_units','sales_revenue'],
               comparison='Paired probe minus student for each identical seed/opponent/seat; aggregate both seats, then opponents within each whole seed. Teacher gap descriptive.',
               exploration_gate='All official/control/freeze gates pass AND probe-minus-student mean net profit > 0 separately in BOTH seeds AND pooled mean margin >= 0 AND pooled match score >= 0 AND market opponent mean profit/margin/score deltas >= 0. Only supports further investigation, never promotion or significance.',
               inference='Two development seeds; no bootstrap/significance/generalization claim. Report every opponent and seed even if gate fails. Changes include downstream market/opponent responses; no causal allocation to individual calls.',
               stop='Exactly one probe, maximum 36 games, no further tuning/seeds or model training. Failed valid games are not replaced; infrastructure failure stops for explicit diagnosis.',
               sources_used_for_selection=['old training','old validation corroboration','old student development corroboration'],
               excluded_for_selection=['old formal test per-game results'],
               acceptance='36 full 720-state/719-action DONE games, no agent errors; official reexecute all ordered states and v1/v2 hashes; both actor actions and all stable traces reproduce; original model/default/teachers/evidence unchanged; zero hidden fallbacks.')
    save(out/'panel-registration.json', cfg)
    oldfreeze = read(source/'freeze.json')['runtime_files']
    paths = {**oldfreeze, **{p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'learning/bc_crop_diagnostics').glob('*.py')}}
    paths['main.py'] = sha(ROOT/'main.py')
    for p in (out/'sources').glob('*.py'): paths[p.relative_to(ROOT).as_posix()] = sha(p)
    save(out/'panel-freeze.json',dict(local_registration_only=True, results_absent=not (out/'panel-ledger.json').exists(),
         registration_sha256=sha(out/'panel-registration.json'), runtime_files=paths,
         original_public_files={p:sha(public/p) for p in tracked},
         analysis_inputs={n:sha(out/n) for n in ['analysis-plan.json','existing-diagnostics.json','overlap-diagnostics.json','diagnostic-decisions-private.jsonl.gz']}))
    print(json.dumps({'prepared':True,'development_seeds':seeds,'game_budget':36}))


def wrapper(out, key, policy, cfg):
    path = out/'agents'/(key+'.py'); path.parent.mkdir(exist_ok=True)
    assert not path.exists() and not (out/'traces'/(key+'.jsonl.gz')).exists()
    source = ('from pathlib import Path\nimport sys\n'
              'ROOT=next(p for p in Path(__file__).resolve().parents if (p/"learning/bc_crop/agent.py").is_file())\n'
              'sys.path.insert(0,str(ROOT))\nfrom learning.bc_crop.agent import CropAgent\n'
              'from learning.bc_crop_diagnostics.probe import StrawberryCorrection\n')
    args = ', trace=ROOT/'+repr((out/'traces'/(key+'.jsonl.gz')).relative_to(ROOT).as_posix())
    if policy=='teacher': source += '_runner=CropAgent("teacher"'+args+')\n'
    else:
        source += '_runner='+('StrawberryCorrection(' if policy=='probe' else 'CropAgent("model",')
        source += 'checkpoint=ROOT/'+repr(cfg['model_relative'])+', expected_sha='+repr(cfg['checkpoint_sha256'])+args+')\n'
    source += 'def agent(observation, configuration):\n    return _runner(observation,configuration)\nsubmission_agent=agent\n'
    path.write_text(source,encoding='utf-8')
    return path


def bounded_jobs(jobs, execute, record, workers):
    """On a failed completion batch, drain in-flight jobs without refilling."""
    pending={}; remaining=iter(jobs); errors=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def refill():
            while len(pending)<workers:
                job=next(remaining,None)
                if job is None:break
                pending[pool.submit(execute,job)]=job
        def consume(futures):
            for future in futures:
                job=pending.pop(future)
                try:record(job,future.result())
                except Exception as exc:errors.append(dict(game=job[0],exception_type=type(exc).__name__))
        refill()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            consume(done)
            if errors:
                # No queued backlog: finish and record already in-flight work.
                consume(as_completed(list(pending)))
                break
            refill()
    return errors


def run_games(out):
    cfg = verify(out)
    assert not (out/'panel-ledger.json').exists() and not (out/'games').exists()
    sys.path.insert(0,str(ROOT/'learning'))
    from optimize_round2 import Games
    todo=[]
    for seed in cfg['development_seeds']:
        for op in cfg['opponents']:
            for side in cfg['seats']:
                for policy in cfg['policies']:
                    key=f'panel-{seed}-{op}-{side}-{policy}'
                    todo.append((key,seed,op,side,policy,wrapper(out,key,policy,cfg)))
    assert len(todo)==cfg['game_budget']==36
    rows=[]; tick=time.perf_counter()
    with Games(out,cfg,'panel') as workers:
        def execute(job):
            key,seed,op,side,policy,bot=job
            return workers.play(bot,policy,op,seed,side,out/'games'/key)
        def record(job,row):
            key,seed,op,side,policy,bot=job
            row.update(game=key,agent_relative=bot.relative_to(ROOT).as_posix(),
                       trace_sha256=sha(out/'traces'/(key+'.jsonl.gz')))
            rows.append(row);save(out/'panel-ledger.json',sorted(rows,key=lambda r:r['game']))
            print(f'development games {len(rows)}/36',flush=True)
        errors=bounded_jobs(todo,execute,record,cfg['workers'])
    if errors:
        save(out/'panel-failure.json',dict(errors=errors,completed=len(rows),budget=36,additional_dispatch_stopped=True))
        raise RuntimeError('Panel stopped after infrastructure failure; retained completed and failed evidence')
    save(out/'panel-timing.json',dict(games=len(rows),wall_seconds=time.perf_counter()-tick))
    verify(out)


def check(job):
    from decision.audit_market_teacher import observation
    from decision.full_audit import reexecute
    out,row=job;out=Path(out);cfg=verify(out);side=row['side']
    raw=out/'games'/row['game']/'replay.json.gz'
    trace=out/'traces'/(row['game']+'.jsonl.gz')
    assert sha(raw)==row['replay_gzip_sha256'] and sha(trace)==row['trace_sha256']
    assert sha(ROOT/row['agent_relative'])==row['bot_sha256']
    with gzip.open(raw,'rt',encoding='utf-8') as f: replay=json.load(f)
    with gzip.open(trace,'rt',encoding='utf-8') as f: turns=[json.loads(l) for l in f]
    assert len(turns)==719
    if row['policy']=='teacher': actor=CropAgent()
    elif row['policy']=='probe': actor=StrawberryCorrection(ROOT/cfg['model_relative'],cfg['checkpoint_sha256'])
    else: actor=CropAgent('model',ROOT/cfg['model_relative'],cfg['checkpoint_sha256'])
    rival=load(out/'sources'/(row['opponent']+'.py')); counts=Counter(); linked={}
    for step,(before,after) in enumerate(zip(replay['steps'],replay['steps'][1:])):
        assert actor(observation(before,side),replay['configuration'])==after[side]['action']
        assert rival(observation(before,1-side),replay['configuration'])==after[1-side]['action']
        assert stable(actor.last)==stable(turns[step])
        assert not any(actor.last[k] for k in ('fallbacks','load_errors','inference_errors'))
        for r in actor.last['records']:
            counts['selector_calls']+=1
            counts['correction_calls']+=r.get('correction_applied',False)
            counts['outcome_'+r['outcome']]+=1
            counts['selected_'+r['chosen']]+=1
            key=(step,r['unit']);assert key not in linked;linked[key]=r
            if r.get('correction_applied'):
                assert r['teacher']==r['chosen']=='STRAWBERRY' and r['model_prediction']!=r['chosen']
            if r['outcome']=='plant_request': assert r['unit_request']==['PLANT',r['chosen']]
    def sink(e):
        if e['player']!=side:return
        counts[e['phase']+'_'+e['status']]+=1
        if e['phase']=='unit' and e.get('successful_plant'):
            record=linked[e['step'],e['unit']]
            assert e['effective']==['PLANT',record['chosen']]
            counts['successful_plants']+=1
            counts['planted_'+record['chosen']]+=1
            counts['corrected_successful_plants']+=record.get('correction_applied',False)
        if e['phase']=='market':
            request=e.get('requested')
            if isinstance(request,list) and request:
                if request[0]=='BUY_SEED':counts['seed_units_bought']+=e.get('executed_units',0)
                if request[0]=='SELL':
                    counts['sales_units']+=e.get('executed_units',0)
                    counts['sales_revenue']+=e.get('cash_delta',0)
    cert=reexecute(replay,trace=True,label_sink=sink)
    assert cert['semantic_sha256_v2']==row['semantic_replay_sha256_v2']
    assert cert['semantic_sha256']==row['semantic_replay_sha256'] and cert['final_rewards']==row['banks']
    final=observation(replay['steps'][-1],side)
    return dict(game=row['game'],policy=row['policy'],seed=row['seed'],opponent=row['opponent'],side=side,
                audit=cert,counts=dict(counts),replay_sha256=sha(raw),trace_sha256=sha(trace),
                reproduced_actor_actions=719,reproduced_rival_actions=719,stable_trace_reproduced=True,
                ending_seed_units=sum(final['private']['seeds'].values()))


def audit(out):
    cfg=verify(out);rows=read(out/'panel-ledger.json')
    assert len(rows)==cfg['game_budget'] and not (out/'panel-audit.json').exists()
    results=[]
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for future in as_completed([pool.submit(check,(str(out),row)) for row in rows]):
            results.append(future.result());print(f'ordered audit {len(results)}/36',flush=True)
    save(out/'panel-audit.json',dict(passed=True,games=len(results),official_transitions=719*len(results),
                                   rows=sorted(results,key=lambda r:r['game'])))
    verify(out)


def analyze(out):
    cfg=verify(out);rows=read(out/'panel-ledger.json');cert=read(out/'panel-audit.json')
    assert cert['passed'] and cert['games']==len(rows)==36
    assert not (out/'panel-results.json').exists()
    by_audit={r['game']:r for r in cert['rows']};paired=[]
    for r in rows:
        r['match_score']=1. if r['margin']>0 else .5 if r['margin']==0 else 0.
        r.update({k:by_audit[r['game']]['counts'].get(k,0) for k in cfg['metrics'] if k in (
            'correction_calls','successful_plants','seed_units_bought','sales_units','sales_revenue')})
        r['ending_seed_units']=by_audit[r['game']]['ending_seed_units']
    for seed in cfg['development_seeds']:
        for op in cfg['opponents']:
            for side in cfg['seats']:
                policies={r['policy']:r for r in rows if (r['seed'],r['opponent'],r['side'])==(seed,op,side)}
                paired.append(dict(seed=seed,opponent=op,side=side,
                    probe_minus_student={k:policies['probe'][k]-policies['student'][k] for k in cfg['metrics']},
                    student_minus_teacher={k:policies['student'][k]-policies['teacher'][k] for k in cfg['metrics']}))
    def delta(part):return {k:mean(r['probe_minus_student'][k] for r in part) for k in cfg['metrics']}
    seed_groups={str(s):delta([r for r in paired if r['seed']==s]) for s in cfg['development_seeds']}
    opponent_groups={op:delta([r for r in paired if r['opponent']==op]) for op in cfg['opponents']}
    overall=delta(paired)
    gate=(all(g['net_profit']>0 for g in seed_groups.values()) and
          all(overall[k]>=0 and opponent_groups['market'][k]>=0 for k in ('margin','match_score')) and
          opponent_groups['market']['net_profit']>=0)
    result=dict(development_only=True,independent_significance_claim=False,exploration_gate_passed=gate,
                new_games=36,training_runs=0,paired=paired,by_seed=seed_groups,by_opponent=opponent_groups,
                probe_minus_student=overall,policies={p:{k:mean(r[k] for r in rows if r['policy']==p) for k in cfg['metrics']} for p in cfg['policies']},
                ledger_sha256=sha(out/'panel-ledger.json'),audit_sha256=sha(out/'panel-audit.json'),
                registration_sha256=sha(out/'panel-registration.json'),freeze_sha256=sha(out/'panel-freeze.json'))
    save(out/'panel-results.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','games','audit','analyze'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--public',type=Path)
    args=p.parse_args();out=args.out.resolve()
    if args.mode=='prepare':prepare(out,args.source.resolve(),args.public.resolve())
    else:{'games':run_games,'audit':audit,'analyze':analyze}[args.mode](out)
