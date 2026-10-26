"""New authorization and seed inventory; frozen anchor validation stays reusable."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil

from learning.task_anchor_once_v1 import experiment as core
from learning.task_anchor_once_v1.statistics import validate_roles
from learning.task_anchor_once_v1.training import TRAINING

ROOT,PUBLIC=core.ROOT,core.PUBLIC
read,sha,write_new,git=core.read,core.sha,core.write_new,core.git
model_identity=core.model_identity
ANCHOR=ROOT/'artifacts/task-anchor-once-v1-20261027'
REPAIR=ROOT/'artifacts/task-anchor-spawn-repair-v1-20261027'
OUT=ROOT/'artifacts/task-anchor-reentry-v1-20261027'


def reentry_authority(out):
    out=Path(out);cfg=read(out/'registration.json');f=read(out/'freeze.json')['reentry']
    assert cfg['experiment']=='task-anchor-reentry-v1'
    assert sha(REPAIR/'root-next-learning-decision.json')==f['decision_sha256']
    assert sha(REPAIR/'root-result-acceptance.json')==f['repair_acceptance_sha256']
    assert sha(REPAIR/'publication.json')==f['repair_publication_sha256']
    assert sha(ANCHOR/'registration.json')==f['method_registration_sha256']
    assert sha(ANCHOR/'root-result-acceptance.json')==f['failed_stage_acceptance_sha256']
    decision=read(REPAIR/'root-next-learning-decision.json')
    assert decision['task']=='task_anchor_reentry_v1' and decision['new_conditional_fit_slot']==1
    assert cfg['training']==TRAINING==decision['training_exact']
    assert cfg['training_seed']==decision['training_rng']
    assert cfg['known_screen_seeds']==decision['known_screen_seeds']
    assert cfg['training_source_seeds']==decision['training_source_seeds']
    for folder,key in [(ANCHOR,'failed_anchor_artifact_sha256'),(REPAIR,'repair_artifact_sha256')]:
        for rel,value in cfg[key].items():assert sha(folder/rel)==value,rel
    return cfg


def verify(out=OUT,history=False):
    core.verify(out,history=history)
    return reentry_authority(out)


def ready(out=OUT):
    verify(out,history=True)
    return core.ready(out)


def training_ready(out):
    cfg=core.training_ready(out)
    reentry_authority(out)
    return cfg


def prepare(out=OUT):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('No continuation or replacement of an existing attempt')
    assert not git('status','--porcelain').strip()
    assert git('rev-parse','HEAD').decode().strip()==read(REPAIR/'publication.json')['commit']
    assert read(REPAIR/'root-result-acceptance.json')['passed']
    decision=read(REPAIR/'root-next-learning-decision.json')
    assert decision['status']=='approved_for_versioned_implementation_not_execution'
    assert decision['task']=='task_anchor_reentry_v1' and decision['new_conditional_fit_slot']==1
    base=core.verify(ANCHOR,history=True)
    cfg=deepcopy(base);frozen=deepcopy(read(ANCHOR/'freeze.json'))
    assert cfg['training']==TRAINING==decision['training_exact']
    assert shutil.disk_usage(ROOT).free>=cfg['storage']['total_bytes']+cfg['storage']['free_floor_bytes']
    tracked=[n for n in git('ls-files','-z').decode().split('\0') if n]
    inventory=core.seed_inventory(tracked);seeds=core.draw_seeds(inventory['excluded_uint32'])
    cache_bytes=cache_rows=0
    for key in ('old_cache','dagger_cache'):
        _hashes,_bytes,_rows=core.cache_inputs(cfg[key])
        assert _hashes[cfg[key]+'/manifest.json']==cfg[key+'_manifest_sha256']
        cache_bytes+=_bytes;cache_rows+=_rows
    assert cache_bytes==cfg['cache_metadata_bytes'] and cache_bytes<=16*1024**3
    assert cache_rows==cfg['cache_metadata_candidate_rows'] and cache_rows<=16000000
    inputs=read(ANCHOR/'reused-cache-inputs.json')
    for rel,value in frozen['runtime_files'].items():assert sha(ROOT/rel)==value,rel
    repair_sources=read(REPAIR/'freeze.json')['new_source_sha256']
    for rel,value in repair_sources.items():assert sha(ROOT/rel)==value,rel
    runtime={**frozen['runtime_files'],**repair_sources}
    runtime.update({p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.iterdir()
                    if p.is_file() and p.suffix in ('.py','.md')})
    public={}
    for rel,value in runtime.items():
        target=PUBLIC/'kaggriculture'/rel
        if target.exists():
            assert core.line_endings_only((ROOT/rel).read_bytes(),target.read_bytes()),rel
            public[rel]=sha(target)
        else:public[rel]=value
    cfg.update(experiment='task-anchor-reentry-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git('rev-parse','HEAD').decode().strip(),runtime_versions=core.versions(),
        explicit_seed_config=seeds,validation_seeds=seeds['validation_seeds'],test_seeds=seeds['test_seeds'],
        family_stop=decision['family_stop'],old_attempt_refund=False,new_fit_slot=1,
        fixed_worker='learning.task_anchor_once_v1.training._worker',
        startup_boundary='Frozen repaired launcher; no globally guarded Process.start',
        fit_attempt_marker='training-launch.json, written before dispatch; failures still consume the slot',
        legacy_method_schema='Frozen worker uses anchor-once schema/family names; registration SHA identifies this new execution',
        previous_goal_turn='progress: a real model-free WSL spawn passed independent acceptance and publication',
        failed_anchor_artifact_sha256={p.relative_to(ANCHOR).as_posix():sha(p) for p in ANCHOR.rglob('*') if p.is_file()},
        repair_artifact_sha256={p.relative_to(REPAIR).as_posix():sha(p) for p in REPAIR.rglob('*') if p.is_file()})
    validate_roles(cfg)
    assert not set(cfg['validation_seeds']+cfg['test_seeds']) & set(base['validation_seeds']+base['test_seeds'])
    out.mkdir(parents=True);(out/'sources').mkdir()
    for name,digest in cfg['source_sha256'].items():
        assert sha(ANCHOR/'sources'/name)==digest
        shutil.copyfile(ANCHOR/'sources'/name,out/'sources'/name)
    shutil.copyfile(ANCHOR/'gpu-launch.local.json',out/'gpu-launch.local.json')
    write_new(out/'seed-inventory.local.json',inventory)
    cfg['seed_inventory_sha256']=sha(out/'seed-inventory.local.json')
    cfg['seed_novelty']=dict(passed=True,public_files=inventory['public_files'],local_files=inventory['local_files'],
        input_bytes=inventory['input_bytes'],excluded_values=len(inventory['excluded_uint32']),
        inventory_sha256=cfg['seed_inventory_sha256'],scope=inventory['scope'],limitation=inventory['limitation'])
    write_new(out/'reused-cache-inputs.json',inputs)
    cfg['reused_cache_inputs_sha256']=sha(out/'reused-cache-inputs.json')
    write_new(out/'registration.json',cfg);write_new(out/'seeds.json',seeds)
    protection=read(ANCHOR/'protection-before.json')['runtime_sha256'].copy();protection.update(runtime)
    history={p.relative_to(ROOT).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
        for p in (ROOT/'artifacts').rglob('*') if p.is_file() and out not in p.parents and '__pycache__' not in p.parts}
    write_new(out/'protection-before.json',dict(history_stat=history,runtime_sha256=protection,
        public_sha256={n:sha(PUBLIC/n) for n in tracked}))
    # Preserve the exact legacy numerical/official-worker authority chain needed
    # by frozen game consumers. Add, rather than substitute, this stage's actual
    # failure->repair->new-slot authorization chain.
    frozen.update(runtime_files=runtime,published_runtime_files=public,
        registration_sha256=sha(out/'registration.json'),seeds_sha256=sha(out/'seeds.json'),
        protection_sha256=sha(out/'protection-before.json'),new_models_and_games_absent=True,
        reentry=dict(decision_sha256=sha(REPAIR/'root-next-learning-decision.json'),
            repair_acceptance_sha256=sha(REPAIR/'root-result-acceptance.json'),
            repair_publication_sha256=sha(REPAIR/'publication.json'),
            method_registration_sha256=sha(ANCHOR/'registration.json'),
            failed_stage_acceptance_sha256=sha(ANCHOR/'root-result-acceptance.json')))
    write_new(out/'freeze.json',frozen)
    return verify(out,history=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','verify'])
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    cfg=prepare(args.out) if args.command=='prepare' else verify(args.out,history=True)
    print(dict(passed=True,experiment=cfg['experiment']))
