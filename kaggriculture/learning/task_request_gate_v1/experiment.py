"""Registration for one whole-request teacher/student reward experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import re
import shutil

from learning.bc_crop.agent import ROOT,sha
from learning.bc_crop.prepare import novelty
from learning.task_bc_holdout_v1.games import write_new
from learning.task_entry_repair_v1.experiment import installed_files
from learning.task_reward_gate_v1.experiment import (PUBLIC,TASK,CROP,TASK_SHA,CROP_SHA,
    ENGINE_SHA,ENGINE_COMMIT,LOADER_SHA,read,git,versions,line_endings_only)

PREVIOUS=ROOT/'artifacts/task-reward-gate-v1-20261027'
OUT=ROOT/'artifacts/task-request-gate-v1-20261027'
LIMITS=dict(task_forwards=4000,crop_forwards=8000,neural_input_rows=160000,
    gate_evaluations=3,controller_calls=839,teacher_proposals=120,
    max_task_forwards_per_turn=32,max_crop_forwards_per_turn=64,
    max_neural_input_rows_per_turn=4096,max_controller_calls_per_turn=2)
HOLDOUT=dict(max_pairs=32,margin_ci_lower_gt=0,opponent_relative_mean_min=-.02,
    opponent_score_delta_min=-.05,minimum_changed_request_games=4,
    teacher_relative_ci_lower_gt=-.02,teacher_opponent_relative_mean_min=-.05,
    teacher_opponent_score_delta_min=-.05)


def seeds():
    values=random.Random(2026102703).sample(range(2**32),18)
    return dict(train_seeds=values[:8],validation_seeds=values[8:10],test_seeds=values[10:],
        generation=dict(rng='Python Random.sample',seed=2026102703,draws=1,population='uint32',count=18))


def prepare(out=OUT):
    out=Path(out)
    if out.exists():raise FileExistsError('Use a new experiment; preserve existing evidence')
    assert not git('status','--porcelain').strip()
    previous=read(PREVIOUS/'root-result-acceptance.json')
    assert previous['collection_and_stop_accepted'] and previous['training_signal_passed'] is False
    assert read(PREVIOUS/'root-next-learning-decision.json')['status']=='approved_for_versioned_implementation_not_execution'
    explicit=seeds();tracked=[n for n in git('ls-files','-z').decode().split('\0') if n]
    scan=novelty({**explicit,'parity_seed':explicit['train_seeds'][0]},PUBLIC,tracked,ROOT/'artifacts')
    values=[s for k in ('train_seeds','validation_seeds','test_seeds') for s in explicit[k]]
    pattern=re.compile(r'(?<!\d)(?:'+'|'.join(map(str,values))+r')(?!\d)')
    attempts=[]
    for p in sorted((ROOT/'artifacts').rglob('*.json')):
        if 'attempt' in p.name.lower() or 'job-inputs' in p.parts:
            assert not pattern.search(p.read_text(encoding='utf-8-sig')),'Previously attempted seed'
            attempts.append([p.relative_to(ROOT).as_posix(),sha(p)])
    import json
    scan['failed_attempt_files']=len(attempts)
    scan['attempt_manifest_sha256']=hashlib.sha256(json.dumps(attempts).encode()).hexdigest()
    official=installed_files()
    assert sha(official['rules'])==ENGINE_SHA and sha(official['loader'])==LOADER_SHA
    assert sha(ROOT/TASK)==TASK_SHA and sha(ROOT/CROP)==CROP_SHA
    runtime=read(PREVIOUS/'freeze.json')['runtime_files'].copy()
    for rel,h in runtime.items():assert sha(ROOT/rel)==h,rel
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ('.py','.md'):runtime[p.relative_to(ROOT).as_posix()]=sha(p)
    published={}
    for rel,h in runtime.items():
        target=PUBLIC/'kaggriculture'/rel
        if target.exists():
            assert line_endings_only((ROOT/rel).read_bytes(),target.read_bytes()),rel
            published[rel]=sha(target)
        else:published[rel]=h
    protected=read(PREVIOUS/'protection-before.json')['runtime_sha256'].copy()
    for rel,h in protected.items():assert sha(ROOT/rel)==h,rel
    protected.update({rel:h for rel,h in runtime.items() if rel.startswith('learning/task_request_gate_v1/')})
    sources={name:sha(PREVIOUS/'sources'/name) for name in ('round2.py','market.py')}
    assert sources==read(PREVIOUS/'registration.json')['source_sha256']
    cfg=dict(experiment='task-request-gate-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git('rev-parse','HEAD').decode().strip(),runtime_versions=versions(),
        previous_experiment=PREVIOUS.relative_to(ROOT).as_posix(),previous_signal_passed=False,
        explicit_seed_config=explicit,**{k:explicit[k] for k in ('train_seeds','validation_seeds','test_seeds')},
        seed_novelty=scan,opponents=['round2','market'],seats=[0,1],
        holdout_arms=['baseline','gated','teacher_task'],windows=[[80,119],[260,299],[440,479]],
        window_assignment='(seed_index+opponent_index+seat)%3',
        task_checkpoint=TASK,task_checkpoint_sha256=TASK_SHA,crop_checkpoint=CROP,crop_checkpoint_sha256=CROP_SHA,
        engine_commit=ENGINE_COMMIT,engine_sha256=ENGINE_SHA,official_loader_sha256=LOADER_SHA,
        source_sha256=sources,max_games=176,max_workers=2,per_job_timeout_seconds=900,campaign_seconds=10800,
        live_transition_limit=126544,primary_reexecution_limit=126544,independent_reexecution_limit=2876,
        total_official_transition_limit=255964,policy_turn_budget=600000,
        task_forward_budget=1200000,crop_forward_budget=3000000,neural_input_row_budget=60000000,
        gate_evaluation_budget=1000,game_limits=LIMITS,
        counter_proof=dict(neural_policy_games=144,full_teacher_task_games=32,
            max_task_forwards=144*2*4000+2*4000,
            max_crop_forwards=176*2*8000+3*8000,
            max_neural_rows=176*2*160000+3*160000,
            max_controllers_and_opponents=176*4*719+2*(80*40+32*120)+3*719,
            explanation='All live and primary calls, teacher proposals and three possible original-controller parity episodes. Full teacher_task has zero task-network calls. Independent replay uses recorded actions, no neural forward. Per-pass caps enforce these worst-case bounds before calls.'),
        early_screen=dict(after_train_seeds=2,planned_cells=8,minimum_eligible=4,any_nonzero_margin_required=True),
        training=dict(max_pairs=32,minimum_eligible=24,minimum_nonzero_margin=12,minimum_nonzero_seeds=6,
            sign_quota=False,fits=1,parameters=9,seconds=60,
            target='(intervention final own-minus-opponent cash - baseline final own-minus-opponent cash)/1000',
            objective='mean squared margin-target error + sum squared non-intercept coefficients',
            normalization='all8 numeric feature columns train-only mean/std with floor1',
            zero_and_negative_targets_retained=True),
        development=dict(max_pairs=8,minimum_eligible=6,minimum_nonzero=4,minimum_nonzero_seeds=2,
            exceeds_constant_student=True,exceeds_constant_teacher_at_opportunity=True,
            each_seed_margin_at_least=0,own_relative_min=-.02,overall_score_delta_min=-.05,tuning=False),
        bootstrap=dict(replicates=20000,seed=2026102704,quantile_method='linear'),holdout=HOLDOUT,
        teacher_definition='TaskAgent(None, model, SAME frozen crop checkpoint); not the original all-rule teacher',
        intervention='Select one complete farmer/hands/ordered-market object at FIRST observed request disagreement in assigned window, then student on all later turns; never blend fields or select by future outcomes.',
        features='step/719; student assignment count; teacher-minus-student selected counts of5OPS; difference of sums of selected task.distance. Aggregate each own trace independently; no teacher priority or ordinal pairing.',
        holdout_behavior='Gated: first disagreement in each of3windows, consume either choice, teacher iff predicted margin advantage>0. Baseline purestudent; teacher_task fullteacher on allturns.',
        missing='Keep missing cell, no intervention game or replacement seed/window. Missing reward target stays unknown; constant-policy no-op effect is zero for dev scoring only.',
        execution_order='Train one complete seed at a time, baseline4cells then eligible interventions; aftersecondseed screen beforeanythirdseedjob. Dev samechronology; holdout fullseed3arms. No outcome-selected extra cells.',
        parity='Firsttrainseed/round2/baseline bothseats reproduce originalJointAgent all719turns; firstholdoutseed/round2/seat0 teacher_task reproduces originalTaskAgent all719turns ifreached.',
        independent_selection='Firsttrainseed/round2/seat0 baseline+intervention ifpresent; firstholdoutseed/market/seat0 gated+teacher_task ifreached. Max4, no replacements.',
        stop='Any execution, source, budget, model identity or audit failure stops dispatch. Single two-seed screen or final signal/dev gate failure closes this whole-request candidate; no retries, gate relaxation, second fit or same-design probe loop.',
        implementation_clock='No human implementation/proof timer; machine campaign10800s, perjob900s, fit60s, independentcompute3600s. Limits are not automatically extended.',
        publication='Only source, explicit registration, aggregate/terminal/pair evidence and hashes; rawReplay/HTML/actions/observations/features/weights/logs/privatepaths remainlocal.',
        local_registration_not_external_timestamp=True,neural_weight_updates=0,gpu_operations=0,
        default_changed=False,automatic_teacher_admission=False)
    # Materialize only after all source/seed prerequisites have passed.
    out.mkdir(parents=True)
    for name in sources:
        (out/'sources').mkdir(exist_ok=True);shutil.copyfile(PREVIOUS/'sources'/name,out/'sources'/name)
    write_new(out/'registration.json',cfg);write_new(out/'seeds.json',explicit)
    history={p.relative_to(ROOT).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
        for p in (ROOT/'artifacts').rglob('*') if p.is_file() and out not in p.parents and '__pycache__' not in p.parts}
    write_new(out/'protection-before.json',dict(history_stat=history,runtime_sha256=protected,
        public_sha256={n:sha(PUBLIC/n) for n in tracked}))
    write_new(out/'freeze.json',dict(runtime_files=runtime,published_runtime_files=published,
        registration_sha256=sha(out/'registration.json'),seeds_sha256=sha(out/'seeds.json'),
        protection_sha256=sha(out/'protection-before.json'),previous_acceptance_sha256=sha(PREVIOUS/'root-result-acceptance.json'),
        new_game_results_absent=True,local_freeze_not_external_timestamp=True))
    return verify(out,history=True)


def gate_identity(out):
    out=Path(out);f=read(out/'gate-model-freeze.json')
    assert f['fit_count']==1 and f['training_pairs_sha256']==sha(out/'train-pairs.json')
    assert sha(out/'gate-model.json')==f['gate_sha256']
    for name in ('fit-result.json','development-gate.json'):
        if (out/name).exists():assert read(out/name)['gate_sha256']==f['gate_sha256']
    return f['gate_sha256']


def verify(out=OUT,history=False):
    out=Path(out);f=read(out/'freeze.json');cfg=read(out/'registration.json')
    assert sha(out/'registration.json')==f['registration_sha256']
    assert sha(out/'seeds.json')==f['seeds_sha256'] and cfg['explicit_seed_config']==read(out/'seeds.json')==seeds()
    assert sha(out/'protection-before.json')==f['protection_sha256']
    assert sha(PREVIOUS/'root-result-acceptance.json')==f['previous_acceptance_sha256']
    assert cfg['runtime_versions']==versions()
    for rel,h in f['runtime_files'].items():assert sha(ROOT/rel)==h,rel
    for name,h in cfg['source_sha256'].items():assert sha(out/'sources'/name)==h
    assert sha(ROOT/TASK)==TASK_SHA and sha(ROOT/CROP)==CROP_SHA
    official=installed_files();assert sha(official['rules'])==ENGINE_SHA and sha(official['loader'])==LOADER_SHA
    if (out/'gate-model-freeze.json').exists():gate_identity(out)
    if history:
        p=read(out/'protection-before.json')
        for rel,h in p['runtime_sha256'].items():assert sha(ROOT/rel)==h,rel
        for rel,identity in p['history_stat'].items():
            st=(ROOT/rel).stat();assert [st.st_size,st.st_mtime_ns]==identity,rel
    return cfg


def ready(out=OUT):
    out=Path(out);cfg=verify(out,history=True);f=read(out/'freeze.json')
    review=read(out/'pre-review.json');receipt=read(out/'pre-execution-commit.json')
    assert review['passed'] and review['freeze_sha256']==sha(out/'freeze.json')
    assert review['registration_sha256']==sha(out/'registration.json')
    for name in ('registration.json','freeze.json','pre-review.json'):
        assert git('show',receipt['commit']+':review/evidence/'+out.name+'/'+name)==(out/name).read_bytes()
    for rel in f['runtime_files']:
        published=git('show',receipt['commit']+':kaggriculture/'+rel)
        assert hashlib.sha256(published).hexdigest()==f['published_runtime_files'][rel]
        assert line_endings_only((ROOT/rel).read_bytes(),published)
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify']);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();cfg=prepare(a.out) if a.command=='prepare' else verify(a.out,history=True)
    print(dict(experiment=cfg['experiment'],passed=True))
