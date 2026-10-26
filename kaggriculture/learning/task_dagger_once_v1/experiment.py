"""One preregistered student-state update with immutable old evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import re
import shutil

from learning.bc_crop.agent import ROOT, sha
from learning.bc_crop.prepare import novelty
from learning.task_bc_holdout_v1.games import write_new
from learning.task_entry_repair_v1.experiment import installed_files
from learning.task_reward_gate_v1.experiment import (PUBLIC, TASK, CROP, TASK_SHA,
    CROP_SHA, ENGINE_SHA, ENGINE_COMMIT, LOADER_SHA, read, git, versions, line_endings_only)
from .policy import LIMITS

PREVIOUS = ROOT / 'artifacts/task-request-gate-v1-20261027'
OUT = ROOT / 'artifacts/task-dagger-once-v1-20261027'
OLD_CACHE = 'artifacts/task-bc-v1-20261023-run1/cache'


def seeds():
    values = random.Random(2026102705).sample(range(2**32), 14)
    return dict(train_seeds=values[:4], validation_seeds=values[4:6], test_seeds=values[6:],
                generation=dict(rng='Python Random.sample', seed=2026102705,
                                draws=1, population='uint32', count=14))


def prepare(out, gpu_config):
    from .training import TRAINING
    from .statistics import ARMS, EARLY, DEVELOPMENT, HOLDOUT, BOOTSTRAP
    out, gpu_config = Path(out).resolve(), Path(gpu_config).resolve()
    if out.exists():
        raise FileExistsError('Preserve existing experiments and failed attempts')
    assert not git('status', '--porcelain').strip()
    decision = read(PREVIOUS/'root-next-learning-decision.json')
    assert decision['task'] == 'task_dagger_once_v1'
    assert decision['status'] == 'approved_for_versioned_implementation_not_execution'
    assert read(PREVIOUS/'root-result-acceptance.json')['passed']
    explicit = seeds()
    tracked = [n for n in git('ls-files', '-z').decode().split('\0') if n]
    scan = novelty({**explicit, 'parity_seed': explicit['train_seeds'][0]},
                   PUBLIC, tracked, ROOT/'artifacts')
    values = [s for k in ('train_seeds', 'validation_seeds', 'test_seeds') for s in explicit[k]]
    pattern = re.compile(r'(?<!\d)(?:'+'|'.join(map(str, values))+r')(?!\d)')
    attempts = []
    for p in sorted((ROOT/'artifacts').rglob('*.json')):
        if 'attempt' in p.name.lower() or 'job-inputs' in p.parts:
            assert not pattern.search(p.read_text(encoding='utf-8-sig')), 'Seed already attempted'
            attempts.append([p.relative_to(ROOT).as_posix(), sha(p)])
    scan.update(failed_attempt_files=len(attempts),
                attempt_manifest_sha256=hashlib.sha256(json.dumps(attempts).encode()).hexdigest())
    official = installed_files()
    assert sha(official['rules']) == ENGINE_SHA and sha(official['loader']) == LOADER_SHA
    assert sha(ROOT/TASK) == TASK_SHA and sha(ROOT/CROP) == CROP_SHA
    cache = read(ROOT/OLD_CACHE/'manifest.json')
    assert cache['passed']
    old_inputs = {OLD_CACHE+'/manifest.json': sha(ROOT/OLD_CACHE/'manifest.json'),
                  OLD_CACHE+'/schema.json': cache['schema_sha256']}
    for split in ('train', 'validation'):
        for name, info in cache['arrays'][split].items():
            assert info['path'] == split+'/'+name+'.npy'
            old_inputs[OLD_CACHE+'/'+info['path']] = info['sha256']
    # Old large arrays are hash-checked once by the trainer, before any fitting.
    # Their size/mtime are protected here; no training feature is decoded.
    for rel in old_inputs:
        assert (ROOT/rel).is_file()
    assert sha(ROOT/OLD_CACHE/'schema.json') == old_inputs[OLD_CACHE+'/schema.json']
    launch = read(gpu_config)
    assert set(launch) == {'distro', 'python', 'workdir', 'expected_device_substring'}
    assert all(type(x) is str and x and '\n' not in x for x in launch.values())
    runtime = read(PREVIOUS/'freeze.json')['runtime_files'].copy()
    for rel, digest in runtime.items():
        assert sha(ROOT/rel) == digest, rel
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ('.py', '.md'):
            runtime[p.relative_to(ROOT).as_posix()] = sha(p)
    published = {}
    for rel, digest in runtime.items():
        target = PUBLIC/'kaggriculture'/rel
        if target.exists():
            assert line_endings_only((ROOT/rel).read_bytes(), target.read_bytes()), rel
            published[rel] = sha(target)
        else:
            published[rel] = digest
    protected = read(PREVIOUS/'protection-before.json')['runtime_sha256'].copy()
    for rel, digest in protected.items():
        assert sha(ROOT/rel) == digest, rel
    protected.update({r:h for r,h in runtime.items() if r.startswith('learning/task_dagger_once_v1/')})
    sources = {n:sha(PREVIOUS/'sources'/n) for n in ('round2.py', 'market.py')}
    assert sources == read(PREVIOUS/'registration.json')['source_sha256']
    cfg = dict(experiment='task-dagger-once-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git('rev-parse', 'HEAD').decode().strip(), runtime_versions=versions(),
        explicit_seed_config=explicit, **{k:explicit[k] for k in ('train_seeds','validation_seeds','test_seeds')},
        seed_novelty=scan, opponents=['round2','market'], seats=[0,1], arms=list(ARMS),
        task_checkpoint=TASK, task_checkpoint_sha256=TASK_SHA,
        crop_checkpoint=CROP, crop_checkpoint_sha256=CROP_SHA,
        old_cache=OLD_CACHE, old_cache_manifest_sha256=old_inputs[OLD_CACHE+'/manifest.json'],
        old_cache_input_sha256=old_inputs, source_sha256=sources,
        engine_commit=ENGINE_COMMIT, engine_sha256=ENGINE_SHA, official_loader_sha256=LOADER_SHA,
        max_games=136, max_workers=2, per_job_timeout_seconds=900, campaign_seconds=10800,
        live_transition_limit=97784, primary_reexecution_limit=97784,
        independent_reexecution_limit=2876, total_official_transition_limit=198444,
        task_forward_budget=776000, crop_forward_budget=2192000,
        neural_input_row_budget=43840000, policy_turn_budget=392574, game_limits=LIMITS,
        counter_proof=dict(neural_policy_games=96, teacher_task_games=40, student_parity_games=2,
            task_forwards=(2*96+2)*4000, crop_forwards=(2*136+2)*8000,
            neural_rows=(2*136+2)*160000, controllers_and_opponents=2*136*2*719+2*719),
        early_screen=dict(EARLY), development=dict(DEVELOPMENT), holdout=dict(HOLDOUT),
        bootstrap=dict(BOOTSTRAP), training=dict(TRAINING), training_seed=2026102707,
        gpu_guard=dict(work_seconds=1185, cleanup_seconds=15, process_seconds=1200),
        data_limits=dict(train_groups=64000, validation_groups=32000,
                         train_candidates=2560000, validation_candidates=1280000),
        gpu_launch_config_sha256=sha(gpu_config),
        gpu_environment=dict(python='3.12.3', versions={'jax':'0.4.38','jaxlib':'0.4.38',
            'numpy':'2.5.4','optax':'0.2.4'}, platform_required='gpu'),
        controller_scope='Frozen teacher opportunities and current student-conditioned reservations; not complete official action control',
        labels='Conditional teacher choice on the same current callback inputs; metadata only, never a deployment feature or reward/optimality claim',
        fit='One512-update warm start with unchanged mean/scale, architecture and crop head; original train+new train equal presentations; one final checkpoint, no selection',
        evaluation='Eight whole-seed clusters; candidate versus both failed original student and hybrid teacher; independent improvement and retention gates',
        parity='First collection seed, round2, baseline both seats: originalJointAgent all719requests',
        independent_selection='Firsttrainseed round2 seat0 baseline; firstdevseed round2 seat0 candidate; firstholdoutseed market seat0 candidate and teacher_task. Missing stages not replaced.',
        stop='Any execution, identity, budget, data, supervision, offline or development failure stops; no retry, refit, replacement seeds/checkpoints or second aggregation iteration',
        privacy='Raw Replay/actions/observations/features/logits/weights/logs/device strings/private launch config stay local; allowlisted summaries only',
        local_registration_not_external_timestamp=True, human_review_stopwatch=False,
        default_changed=False, automatic_teacher_admission=False)
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    for name in sources:
        shutil.copyfile(PREVIOUS/'sources'/name, out/'sources'/name)
    shutil.copyfile(gpu_config, out/'gpu-launch.local.json')
    write_new(out/'registration.json', cfg)
    write_new(out/'seeds.json', explicit)
    history = {p.relative_to(ROOT).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
        for p in (ROOT/'artifacts').rglob('*') if p.is_file() and out not in p.parents and '__pycache__' not in p.parts}
    write_new(out/'protection-before.json', dict(history_stat=history,
        runtime_sha256=protected, public_sha256={n:sha(PUBLIC/n) for n in tracked}))
    write_new(out/'freeze.json', dict(runtime_files=runtime, published_runtime_files=published,
        registration_sha256=sha(out/'registration.json'), seeds_sha256=sha(out/'seeds.json'),
        protection_sha256=sha(out/'protection-before.json'),
        previous_acceptance_sha256=sha(PREVIOUS/'root-result-acceptance.json'),
        decision_sha256=sha(PREVIOUS/'root-next-learning-decision.json'),
        new_game_results_absent=True, local_freeze_not_external_timestamp=True))
    return verify(out, history=True)


def model_identity(out):
    out = Path(out)
    value = read(out/'model-freeze.json')
    assert value['fit_count'] == 1 and value['updates'] == 512
    assert value['registration_sha256'] == sha(out/'registration.json')
    assert value['cache_manifest_sha256'] == sha(out/'cache/manifest.json')
    assert value['initial_task_sha256'] == TASK_SHA
    assert value['checkpoint_sha256'] == sha(out/'model/trained.npz')
    return value['checkpoint_sha256']


def source_verify(out):
    out = Path(out)
    f, cfg = read(out/'freeze.json'), read(out/'registration.json')
    assert sha(out/'registration.json') == f['registration_sha256']
    assert sha(out/'seeds.json') == f['seeds_sha256'] and cfg['explicit_seed_config'] == read(out/'seeds.json') == seeds()
    assert sha(out/'protection-before.json') == f['protection_sha256']
    assert sha(PREVIOUS/'root-result-acceptance.json') == f['previous_acceptance_sha256']
    assert sha(PREVIOUS/'root-next-learning-decision.json') == f['decision_sha256']
    for rel, digest in f['runtime_files'].items():
        assert sha(ROOT/rel) == digest, rel
    for name, digest in cfg['source_sha256'].items():
        assert sha(out/'sources'/name) == digest, name
    assert sha(ROOT/TASK) == TASK_SHA and sha(ROOT/CROP) == CROP_SHA
    assert sha(ROOT/cfg['old_cache']/'manifest.json') == cfg['old_cache_manifest_sha256']
    assert sha(out/'gpu-launch.local.json') == cfg['gpu_launch_config_sha256']
    if (out/'model-freeze.json').exists():
        model_identity(out)
    return cfg


def verify(out=OUT, history=False):
    out = Path(out)
    cfg = source_verify(out)
    assert cfg['runtime_versions'] == versions()
    official = installed_files()
    assert sha(official['rules']) == ENGINE_SHA and sha(official['loader']) == LOADER_SHA
    if history:
        old = read(out/'protection-before.json')
        for rel, digest in old['runtime_sha256'].items():
            assert sha(ROOT/rel) == digest, rel
        for rel, identity in old['history_stat'].items():
            st = (ROOT/rel).stat()
            assert [st.st_size,st.st_mtime_ns] == identity, rel
    return cfg


def ready(out=OUT):
    out = Path(out)
    cfg, f = verify(out, history=True), read(out/'freeze.json')
    review, receipt = read(out/'pre-review.json'), read(out/'pre-execution-commit.json')
    assert review['passed'] and review['freeze_sha256'] == sha(out/'freeze.json')
    assert review['registration_sha256'] == sha(out/'registration.json')
    for name in ('registration.json','freeze.json','pre-review.json'):
        assert git('show',receipt['commit']+':review/evidence/'+out.name+'/'+name) == (out/name).read_bytes()
    for rel, digest in f['published_runtime_files'].items():
        published = git('show',receipt['commit']+':kaggriculture/'+rel)
        assert hashlib.sha256(published).hexdigest() == digest
        assert line_endings_only((ROOT/rel).read_bytes(),published)
    return cfg


def training_ready(out):
    """WSL source/data identity, without requiring Windows runtime versions."""
    out = Path(out)
    cfg = source_verify(out)
    launch = read(out/'training-launch.json')
    for name in ('registration.json','freeze.json','pre-execution-commit.json','cache/manifest.json'):
        assert launch['input_sha256'][name] == sha(out/name), name
    assert launch['fit_count'] == 1 and read(out/'early-gate.json')['passed']
    expected = cfg['gpu_environment']
    assert platform.python_version() == expected['python']
    assert {k:version(k) for k in expected['versions']} == expected['versions']
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','verify'])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--gpu-config', type=Path)
    args = parser.parse_args()
    cfg = prepare(args.out,args.gpu_config) if args.command == 'prepare' else verify(args.out,history=True)
    print(dict(passed=True,experiment=cfg['experiment']))
