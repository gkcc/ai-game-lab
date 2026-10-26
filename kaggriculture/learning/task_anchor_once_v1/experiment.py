"""Explicit reused-data roles and fresh game seeds for one fixed anchor control."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import shutil

import numpy as np
from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.prepare import TEXT_SUFFIXES
from learning.task_bc_holdout_v1.games import write_new
from learning.task_entry_repair_v1.experiment import installed_files
from learning.task_reward_gate_v1.experiment import (
    PUBLIC, TASK, CROP, TASK_SHA, CROP_SHA, ENGINE_SHA, ENGINE_COMMIT,
    LOADER_SHA, read, git, versions, line_endings_only)
from learning.task_dagger_once_v1.policy import LIMITS

PREVIOUS = ROOT/'artifacts/task-dagger-once-v1-20261027'
OUT = ROOT/'artifacts/task-anchor-once-v1-20261027'
OLD_CACHE = 'artifacts/task-bc-v1-20261023-run1/cache'
DAGGER_CACHE = 'artifacts/task-dagger-once-v1-20261027/cache'
MIB = 1024**2
STORAGE = dict(total_bytes=16384*MIB, game_write_bytes=128*MIB,
    fit_write_bytes=512*MIB, primary_parent_write_bytes=256*MIB,
    independent_reserve_bytes=256*MIB, free_floor_bytes=4096*MIB,
    process_log_bytes=MIB, small_receipt_bytes=16384,
    training_parent_write_bytes=65536, supervisor_write_bytes=65536)


def draw_seeds(excluded):
    rng = np.random.Generator(np.random.PCG64(2026102718))
    chosen, rejected = [], []
    excluded = set(excluded)
    while len(chosen) < 10:
        value = int(rng.integers(0, 2**32))
        if value in excluded or value in chosen:
            rejected.append(value)
        else:
            chosen.append(value)
    return dict(validation_seeds=chosen[:2], test_seeds=chosen[2:],
        generation=dict(rng='PCG64', seed=2026102718, population='uint32',
                        draws=10+len(rejected), rejected=rejected,
                        rule='First ten eligible values; only inventory matches and duplicates rejected'))


def seed_inventory(tracked):
    """Same bounded metadata scope as prior novelty, plus attempts and job inputs."""
    files = [('public/'+n,PUBLIC/n) for n in tracked if Path(n).suffix.lower() in TEXT_SUFFIXES]
    local = sorted(p for p in (ROOT/'artifacts').rglob('*.json')
        if not p.name.lower().startswith('replay') and
        (re.search(r'(registration|manifest|ledger|config|result|summary|seed|attempt)',p.name,re.I)
         or 'job-inputs' in p.parts))
    files += [('local/'+p.relative_to(ROOT/'artifacts').as_posix(),p) for p in local]
    found, identities, total = set(), [], 0
    pattern = re.compile(r'(?<!\d)\d{1,10}(?!\d)')
    for label,path in files:
        raw = path.read_bytes(); total += len(raw)
        found.update(v for m in pattern.finditer(raw.decode('utf-8-sig')) if (v:=int(m.group())) < 2**32)
        identities.append([label,hashlib.sha256(raw).hexdigest()])
    return dict(excluded_uint32=sorted(found), files=identities,
        public_files=len(files)-len(local), local_files=len(local), input_bytes=total,
        scope='Tracked public text and existing seed-bearing JSON metadata including attempts/job inputs; no Replay or numeric arrays',
        limitation='Conservative decimal occurrences, not uniqueness against deleted/unavailable/external history')


def cache_inputs(relative):
    folder = ROOT/relative
    manifest = read(folder/'manifest.json')
    assert manifest['passed'] and sha(folder/'schema.json') == manifest['schema_sha256']
    hashes = {relative+'/manifest.json':sha(folder/'manifest.json'),
              relative+'/schema.json':manifest['schema_sha256']}
    size = candidates = 0
    for split, arrays in manifest['arrays'].items():
        assert split in ('train','validation')
        for name,item in arrays.items():
            assert item['path'] == split+'/'+name+'.npy'
            path = folder/item['path']; assert path.is_file()
            size += path.stat().st_size
            hashes[relative+'/'+item['path']] = item['sha256']
        candidates += arrays['candidates']['shape'][0]
    return hashes,size,candidates


def prepare(out, gpu_config):
    from .training import TRAINING
    from .statistics import ARMS, DEVELOPMENT, HOLDOUT, BOOTSTRAP, validate_roles
    out,gpu_config = Path(out).resolve(),Path(gpu_config).resolve()
    if out.exists():raise FileExistsError('Keep all completed and failed runs; no resumption')
    assert not git('status','--porcelain').strip()
    decision = read(PREVIOUS/'root-next-learning-decision.json')
    assert decision['task']=='task_anchor_once_v1' and decision['status']=='approved_for_versioned_implementation_not_execution'
    assert read(PREVIOUS/'root-result-acceptance.json')['passed']
    assert read(PREVIOUS/'publication.json')['commit']==git('rev-parse','HEAD').decode().strip()
    tracked=[n for n in git('ls-files','-z').decode().split('\0') if n]
    inventory=seed_inventory(tracked); explicit=draw_seeds(inventory['excluded_uint32'])
    previous=read(PREVIOUS/'registration.json')
    official=installed_files()
    assert sha(official['rules'])==ENGINE_SHA and sha(official['loader'])==LOADER_SHA
    assert sha(ROOT/TASK)==TASK_SHA and sha(ROOT/CROP)==CROP_SHA
    old,old_size,old_rows=cache_inputs(OLD_CACHE)
    dagger,dagger_size,dagger_rows=cache_inputs(DAGGER_CACHE)
    advice=read(PREVIOUS/'next-learning-advice.json')
    assert old[OLD_CACHE+'/manifest.json']==advice['reuse']['old_manifest']
    assert dagger[DAGGER_CACHE+'/manifest.json']==advice['reuse']['dagger_manifest']
    assert old_size+dagger_size <= 16*1024**3 and old_rows+dagger_rows <=16000000
    assert shutil.disk_usage(ROOT).free >= STORAGE['total_bytes']+STORAGE['free_floor_bytes']
    launch=read(gpu_config)
    assert set(launch)=={'distro','python','workdir','expected_device_substring'}
    assert all(type(v) is str and v.strip() and '\n' not in v for v in launch.values())
    runtime=read(PREVIOUS/'freeze.json')['runtime_files'].copy()
    for rel,digest in runtime.items():assert sha(ROOT/rel)==digest,rel
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ('.py','.md'):runtime[p.relative_to(ROOT).as_posix()]=sha(p)
    published={}
    for rel,digest in runtime.items():
        target=PUBLIC/'kaggriculture'/rel
        if target.exists():
            assert line_endings_only((ROOT/rel).read_bytes(),target.read_bytes()),rel
            published[rel]=sha(target)
        else:published[rel]=digest
    protected=read(PREVIOUS/'protection-before.json')['runtime_sha256'].copy()
    for rel,digest in protected.items():assert sha(ROOT/rel)==digest,rel
    protected.update({r:h for r,h in runtime.items() if r.startswith('learning/task_anchor_once_v1/')})
    sources=previous['source_sha256']
    for n,digest in sources.items():assert sha(PREVIOUS/'sources'/n)==digest
    cfg=dict(experiment='task-anchor-once-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git('rev-parse','HEAD').decode().strip(),runtime_versions=versions(),
        explicit_seed_config=explicit,validation_seeds=explicit['validation_seeds'],test_seeds=explicit['test_seeds'],
        training_source_seeds=previous['train_seeds'],known_screen_seeds=previous['validation_seeds'],
        seed_roles='Only validation_seeds/test_seeds are fresh official game seeds; reused cache screens informed design',
        known_screens_independent=False,opponents=['round2','market'],seats=[0,1],arms=list(ARMS),
        task_checkpoint=TASK,task_checkpoint_sha256=TASK_SHA,crop_checkpoint=CROP,crop_checkpoint_sha256=CROP_SHA,
        old_cache=OLD_CACHE,old_cache_manifest_sha256=old[OLD_CACHE+'/manifest.json'],
        dagger_cache=DAGGER_CACHE,dagger_cache_manifest_sha256=dagger[DAGGER_CACHE+'/manifest.json'],
        cache_metadata_bytes=old_size+dagger_size,cache_metadata_candidate_rows=old_rows+dagger_rows,
        source_sha256=sources,engine_commit=ENGINE_COMMIT,engine_sha256=ENGINE_SHA,official_loader_sha256=LOADER_SHA,
        max_games=120,max_workers=2,per_job_timeout_seconds=900,campaign_seconds=10800,
        live_transition_limit=86280,primary_reexecution_limit=86280,independent_reexecution_limit=2157,
        total_official_transition_limit=174717,task_forward_budget=640000,crop_forward_budget=1920000,
        neural_input_row_budget=38400000,policy_turn_budget=345120,game_limits=LIMITS,
        counter_proof=dict(neural_policy_games=80,teacher_task_games=40,extra_shadow_games=0,
            task_forwards=2*80*4000,crop_forwards=2*120*8000,neural_rows=2*120*160000,
            controllers_and_opponents=2*120*2*719),
        storage=dict(STORAGE),development=dict(DEVELOPMENT),holdout=dict(HOLDOUT),bootstrap=dict(BOOTSTRAP),
        training=dict(TRAINING),training_seed=2026102707,
        gpu_guard=dict(work_seconds=1185,cleanup_seconds=15,process_seconds=1200),
        gpu_launch_config_sha256=sha(gpu_config),gpu_environment=previous['gpu_environment'],
        fit='One original-S update with fixed old-distribution KL coefficient1; failed previous candidate never loaded',
        family_stop=decision['family_stop_rule'],default_changed=False,automatic_teacher_admission=False,
        independent_selection='First fresh dev seed round2 seat0 C; first fresh holdout seed market seat0 C and T; no replacement',
        output_scope='Guarded Python artifact writers; native stdout/stderr captured through bounded parent pipes. No claim to sandbox arbitrary native filesystem syscalls.',
        log_adapter='Reuse frozen run_match.main; only native_output FD redirection is disabled because native output already has a bounded parent pipe. Python run.log remains metered.',
        privacy='Raw Replay, states, actions, traces, features, logits, labels, weights, private device/config/logs stay local',
        local_registration_not_external_timestamp=True)
    validate_roles(cfg)
    out.mkdir(parents=True);(out/'sources').mkdir()
    for name in sources:shutil.copyfile(PREVIOUS/'sources'/name,out/'sources'/name)
    shutil.copyfile(gpu_config,out/'gpu-launch.local.json')
    write_new(out/'seed-inventory.local.json',inventory)
    cfg['seed_inventory_sha256']=sha(out/'seed-inventory.local.json')
    cfg['seed_novelty']=dict(passed=True,public_files=inventory['public_files'],local_files=inventory['local_files'],
        input_bytes=inventory['input_bytes'],excluded_values=len(inventory['excluded_uint32']),
        inventory_sha256=cfg['seed_inventory_sha256'],scope=inventory['scope'],limitation=inventory['limitation'])
    write_new(out/'reused-cache-inputs.json',dict(input_sha256={**old,**dagger},
        old_manifest_sha256=cfg['old_cache_manifest_sha256'],dagger_manifest_sha256=cfg['dagger_cache_manifest_sha256'],
        training_source_seeds=cfg['training_source_seeds'],known_screen_seeds=cfg['known_screen_seeds'],
        array_hash_verification_deferred_until_fit=True,arrays_read_during_prepare=False))
    cfg['reused_cache_inputs_sha256']=sha(out/'reused-cache-inputs.json')
    write_new(out/'registration.json',cfg);write_new(out/'seeds.json',explicit)
    history={p.relative_to(ROOT).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
        for p in (ROOT/'artifacts').rglob('*') if p.is_file() and out not in p.parents and '__pycache__' not in p.parts}
    write_new(out/'protection-before.json',dict(history_stat=history,runtime_sha256=protected,
                                               public_sha256={n:sha(PUBLIC/n) for n in tracked}))
    write_new(out/'freeze.json',dict(runtime_files=runtime,published_runtime_files=published,
        registration_sha256=sha(out/'registration.json'),seeds_sha256=sha(out/'seeds.json'),
        protection_sha256=sha(out/'protection-before.json'),previous_acceptance_sha256=sha(PREVIOUS/'root-result-acceptance.json'),
        decision_sha256=sha(PREVIOUS/'root-next-learning-decision.json'),advice_sha256=sha(PREVIOUS/'next-learning-advice.json'),
        new_models_and_games_absent=True,local_freeze_not_external_timestamp=True))
    return verify(out,history=True)


def model_identity(out):
    out=Path(out);value=read(out/'model-freeze.json');cfg=read(out/'registration.json')
    assert value['fit_count']==1 and value['updates']==512
    assert value['registration_sha256']==sha(out/'registration.json')
    assert value['cache_manifest_sha256']==cfg['dagger_cache_manifest_sha256']==sha(ROOT/cfg['dagger_cache']/'manifest.json')
    assert value['initial_task_sha256']==TASK_SHA and value['checkpoint_sha256']==sha(out/'model/trained.npz')
    return value['checkpoint_sha256']


def source_verify(out):
    from .statistics import validate_roles
    out=Path(out);f,cfg=read(out/'freeze.json'),read(out/'registration.json')
    for name in ('registration','seeds','protection-before'):
        key='protection_sha256' if name=='protection-before' else name+'_sha256'
        assert sha(out/(name+'.json'))==f[key]
    assert cfg['explicit_seed_config']==read(out/'seeds.json')
    assert sha(out/'seed-inventory.local.json')==cfg['seed_inventory_sha256']
    assert draw_seeds(read(out/'seed-inventory.local.json')['excluded_uint32'])==cfg['explicit_seed_config']
    validate_roles(cfg)
    assert sha(PREVIOUS/'root-result-acceptance.json')==f['previous_acceptance_sha256']
    assert sha(PREVIOUS/'root-next-learning-decision.json')==f['decision_sha256']
    assert sha(PREVIOUS/'next-learning-advice.json')==f['advice_sha256']
    for rel,digest in f['runtime_files'].items():assert sha(ROOT/rel)==digest,rel
    for name,digest in cfg['source_sha256'].items():assert sha(out/'sources'/name)==digest,name
    assert sha(ROOT/TASK)==TASK_SHA and sha(ROOT/CROP)==CROP_SHA
    assert sha(out/'reused-cache-inputs.json')==cfg['reused_cache_inputs_sha256']
    for key in ('old_cache','dagger_cache'):
        assert sha(ROOT/cfg[key]/'manifest.json')==cfg[key+'_manifest_sha256']
    assert sha(out/'gpu-launch.local.json')==cfg['gpu_launch_config_sha256']
    if (out/'model-freeze.json').exists():model_identity(out)
    return cfg


def verify(out=OUT,history=False):
    out=Path(out);cfg=source_verify(out)
    assert cfg['runtime_versions']==versions()
    official=installed_files()
    assert sha(official['rules'])==ENGINE_SHA and sha(official['loader'])==LOADER_SHA
    if history:
        old=read(out/'protection-before.json')
        for rel,digest in old['runtime_sha256'].items():assert sha(ROOT/rel)==digest,rel
        for rel,identity in old['history_stat'].items():
            st=(ROOT/rel).stat();assert [st.st_size,st.st_mtime_ns]==identity,rel
    return cfg


def ready(out=OUT):
    out=Path(out);cfg=verify(out,history=True);f=read(out/'freeze.json')
    review,receipt=read(out/'pre-review.json'),read(out/'pre-execution-commit.json')
    assert review['passed'] and review['freeze_sha256']==sha(out/'freeze.json')
    assert review['registration_sha256']==sha(out/'registration.json')
    for name in ('registration.json','freeze.json','pre-review.json'):
        assert git('show',receipt['commit']+':review/evidence/'+out.name+'/'+name)==(out/name).read_bytes()
    for rel,digest in f['published_runtime_files'].items():
        published=git('show',receipt['commit']+':kaggriculture/'+rel)
        assert hashlib.sha256(published).hexdigest()==digest
        assert line_endings_only((ROOT/rel).read_bytes(),published)
    return cfg


def training_ready(out):
    out=Path(out);cfg=source_verify(out);launch=read(out/'training-launch.json')
    for name in ('registration.json','freeze.json','pre-execution-commit.json','reused-cache-inputs.json'):
        assert launch['input_sha256'][name]==sha(out/name),name
    assert launch['fit_count']==1
    expected=cfg['gpu_environment']
    assert platform.python_version()==expected['python']
    assert {k:version(k) for k in expected['versions']}==expected['versions']
    return cfg


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','verify'])
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--gpu-config',type=Path)
    args=parser.parse_args()
    cfg=prepare(args.out,args.gpu_config) if args.command=='prepare' else verify(args.out,history=True)
    print(dict(passed=True,experiment=cfg['experiment']))
