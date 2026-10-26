"""One registered reward-gate experiment; old experiments remain immutable."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
from pathlib import Path
import random
import re
import shutil
import subprocess

from learning.bc_crop.agent import ROOT, sha, TEACHER_SHA
from learning.bc_crop.prepare import novelty
from learning.task_entry_repair_v1.experiment import installed_files
from learning.task_bc_holdout_v1.games import write_new

PUBLIC = ROOT.parent / 'publication/ai-game-lab'
OUT = ROOT / 'artifacts/task-reward-gate-v1-20261027'
TASK = 'artifacts/task-bc-holdout-v1-20261023/model/trained.npz'
CROP = 'artifacts/bc-epoch6-holdout-v1-20261023/panels/fixed_epoch6/model/trained.npz'
TASK_SHA = '9253bc05e632eec2d5a813557272655ab97e023c09843c60a80781b6cc2ab143'
CROP_SHA = 'd497e8284dde1f7b836b1cf88915822f82a96be1f99e7b126d06ea263e281ffd'
ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
LOADER_SHA = '0ac7baf1212b4ce1eabae3cc1cbb55c98effef5e0d0485522a8be87097c39a4f'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git(*args):
    return subprocess.check_output(['git', '-C', str(PUBLIC), *args])


def line_endings_only(runtime, published):
    """Old published source has four documented CRLF/LF conversions only."""
    return runtime.replace(b'\r\n',b'\n')==published.replace(b'\r\n',b'\n')


def versions():
    return dict(python=platform.python_version(), numpy=version('numpy'),
                kaggle_environments=version('kaggle-environments'))


def seeds():
    values = random.Random(2026102701).sample(range(2**32), 18)
    return dict(train_seeds=values[:8], validation_seeds=values[8:10], test_seeds=values[10:],
                generation=dict(rng='Python Random.sample', seed=2026102701,
                                draws=1, population='uint32', count=18))


def prepare(out=OUT):
    out = Path(out)
    assert sorted(p.name for p in out.iterdir()) == ['implementation-start.json']
    assert not git('status', '--porcelain').strip()
    explicit = seeds()
    tracked = [n for n in git('ls-files', '-z').decode().split('\0') if n]
    scan = novelty({**explicit, 'parity_seed': explicit['train_seeds'][0]},
                   PUBLIC, tracked, ROOT / 'artifacts')
    # Failed attempts are consumed even if no replay/result was ever written.
    values = [s for k in ('train_seeds', 'validation_seeds', 'test_seeds') for s in explicit[k]]
    pattern = re.compile(r'(?<!\d)(?:' + '|'.join(map(str, values)) + r')(?!\d)')
    failed = []
    for p in sorted((ROOT / 'artifacts').rglob('*.json')):
        if ('attempt' in p.name.lower() or 'job-inputs' in p.parts) and out not in p.parents:
            raw = p.read_bytes()
            assert not pattern.search(raw.decode('utf-8-sig')), 'Seed used in old attempt'
            failed.append([p.relative_to(ROOT).as_posix(), sha(p)])
    import hashlib
    scan['failed_attempt_files'] = len(failed)
    scan['failed_attempt_manifest_sha256'] = hashlib.sha256(json.dumps(failed).encode()).hexdigest()
    official = installed_files()
    assert sha(official['rules']) == ENGINE_SHA and sha(official['loader']) == LOADER_SHA
    assert sha(ROOT / TASK) == TASK_SHA and sha(ROOT / CROP) == CROP_SHA
    assert sha(ROOT / 'bots/teacher_fertilizer_terminal_v2.py') == TEACHER_SHA
    originals = ROOT / 'artifacts/bc-crop-v1-20261023-run1'
    original_cfg = read(originals / 'registration.json')
    source_hashes = {}
    for name in ('round2', 'market'):
        src = originals / 'sources' / (name + '.py')
        assert sha(src) == original_cfg['source_sha256'][name + '.py']
        target = out / 'sources' / src.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, target)
        source_hashes[src.name] = sha(target)
    cfg = dict(experiment='task-reward-gate-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git('rev-parse', 'HEAD').decode().strip(), runtime_versions=versions(),
        explicit_seed_config=explicit, **{k: explicit[k] for k in ('train_seeds','validation_seeds','test_seeds')},
        seed_novelty=scan, opponents=['round2','market'], seats=[0,1],
        windows=[[80,119],[260,299],[440,479]], window_assignment='(seed_index+opponent_index+seat)%3',
        task_checkpoint=TASK, task_checkpoint_sha256=TASK_SHA,
        crop_checkpoint=CROP, crop_checkpoint_sha256=CROP_SHA, source_sha256=source_hashes,
        engine_commit=ENGINE_COMMIT, engine_sha256=ENGINE_SHA, official_loader_sha256=LOADER_SHA,
        max_workers=2, per_job_timeout_seconds=900, campaign_seconds=5400,
        max_games=144, live_transition_limit=103536, primary_reexecution_limit=103536,
        independent_reexecution_limit=2876, total_official_transition_limit=209948,
        independent_reexecution_selection='first two train cells, baseline and intervention if present; at most four games; no outcome selection',
        policy_turn_budget=416000, task_forward_budget=1200000, crop_forward_budget=1200000,
        neural_input_row_budget=42000000, gate_evaluation_budget=2000,
        game_limits=dict(task_forwards=4000, crop_forwards=4000, neural_input_rows=140000,
                         max_task_forwards_per_turn=32,max_crop_forwards_per_turn=32,
                         max_neural_input_rows_per_turn=2000,gate_evaluations=3),
        shadow_limits=dict(task_forwards=32, crop_forwards=32, neural_input_rows=2000),
        training=dict(max_pairs=32, minimum_eligible=24, minimum_nonzero=12, minimum_positive=4,
                      minimum_negative=4, minimum_nonzero_seeds=6, fits=1, parameters=13,
                      objective='mean squared target error + 1.0*sum(nonintercept weights squared)',
                      target='(intervention own terminal cash - baseline own terminal cash)/1000',
                      continuous_normalization='train only mean/std with scale floor1; other ten onehot columns unchanged',
                      seconds=60, zero_negative_and_same_action_pairs_retained=True),
        development=dict(max_pairs=8, minimum_eligible=6, minimum_nonzero=4,
                         minimum_nonzero_seeds=2, minimum_positive_choices=2,
                         mean_realized_uplift_gt=0, each_seed_mean_at_least=0, tuning=False),
        bootstrap=dict(replicates=20000, seed=2026102702, quantile_method='linear'),
        holdout=dict(max_pairs=32, relative_ci_lower_gt=0, opponent_relative_mean_min=-.02,
                     opponent_score_delta_min=-.05, minimum_changed_request_games=4),
        baseline='frozen task+crop student; prior independent panel failed retention; not teacher_task',
        continuation='one top2 intervention then same frozen student including same-turn suffix; holdout up to3 consumed opportunities',
        candidate_scope='existing frozen teacher economic opportunities, not complete official legal action space',
        train_or_dev_missing='retain cell with eligible=false; no alternative B game, replacement, extra window or seed',
        stop='Any source/entry/audit/counter/timeout failure stops dispatch. No retries or replacement. Train signal or dev gate failure stops before later stages.',
        accounting='Live, primary reproduction, same-state shadow, original student parity and independent official-only replay counted separately; crop row means one input vector, not five output logits.',
        neural_weight_updates=0, gpu_operations=0, default_changed=False, teacher_admission=False,
        interpretation='Restricted reward selector research on two known opponents; local preregistration is not a third-party timestamp; no teacher superiority claim.')
    write_new(out / 'seeds.json', explicit)
    write_new(out / 'registration.json', cfg)
    return freeze(out)


def freeze(out=OUT):
    """Finish a preparation without replacing its seeds or registration.

    This does not authorize games. A partial preparation can be completed only
    before any attempts, protection manifest, freeze or pre-execution commit.
    """
    out=Path(out)
    for name in ('freeze.json','protection-before.json','campaign-started.json','job-inputs','pre-execution-commit.json'):
        assert not (out/name).exists(),name
    cfg=read(out/'registration.json')
    assert cfg['explicit_seed_config']==read(out/'seeds.json')==seeds()
    assert cfg['runtime_versions']==versions()
    assert cfg['baseline_head']==git('rev-parse','HEAD').decode().strip()
    assert not git('status','--porcelain').strip()
    tracked=[n for n in git('ls-files','-z').decode().split('\0') if n]
    runtime = read(ROOT/'artifacts/task-bc-holdout-v1-20261023/freeze.json')['runtime_files'].copy()
    for rel,h in runtime.items(): assert sha(ROOT/rel)==h, rel
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ('.py','.md'):runtime[p.relative_to(ROOT).as_posix()]=sha(p)
    protected = {}
    for base in (ROOT/'learning', ROOT/'decision', ROOT/'bots'):
        for p in sorted(base.rglob('*.py')):
            protected[p.relative_to(ROOT).as_posix()] = sha(p)
    for name in ('main.py','run_match.py','learning/official_worker.py'):
        runtime[name] = protected[name] = sha(ROOT/name)
    # Old raw artifacts are large. Preserve a stat inventory plus explicit original
    # public bytes/source/model hashes; don't call this a full history rehash.
    history = {p.relative_to(ROOT).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
               for p in (ROOT/'artifacts').rglob('*')
               if p.is_file() and out not in p.parents and '__pycache__' not in p.parts}
    write_new(out/'protection-before.json', dict(history_stat=history,
        public_sha256={n:sha(PUBLIC/n) for n in tracked}, runtime_sha256=protected))
    write_new(out/'freeze.json', dict(runtime_files=runtime, registration_sha256=sha(out/'registration.json'),
        seeds_sha256=sha(out/'seeds.json'), protection_sha256=sha(out/'protection-before.json'),
        published_runtime_files={rel:sha(PUBLIC/'kaggriculture'/rel) if (PUBLIC/'kaggriculture'/rel).exists() else h
                                 for rel,h in runtime.items()},
        local_freeze_not_external_timestamp=True, new_game_results_absent=True))
    return verify(out)


def verify(out=OUT, history=False):
    out = Path(out); cfg = read(out/'registration.json'); f = read(out/'freeze.json')
    assert sha(out/'registration.json') == f['registration_sha256']
    assert sha(out/'seeds.json') == f['seeds_sha256']
    assert sha(out/'protection-before.json') == f['protection_sha256']
    assert cfg['explicit_seed_config'] == seeds() and cfg['runtime_versions'] == versions()
    for rel, h in f['runtime_files'].items():
        assert sha(ROOT/rel) == h, rel
    for name,h in cfg['source_sha256'].items(): assert sha(out/'sources'/name)==h
    assert sha(ROOT/TASK)==TASK_SHA and sha(ROOT/CROP)==CROP_SHA
    if (out/'gate-model-freeze.json').exists():
        gate_identity(out)
    official=installed_files()
    assert sha(official['rules'])==ENGINE_SHA and sha(official['loader'])==LOADER_SHA
    if history:
        for rel,h in read(out/'protection-before.json')['runtime_sha256'].items():
            assert sha(ROOT/rel)==h,rel
        for rel, identity in read(out/'protection-before.json')['history_stat'].items():
            st=(ROOT/rel).stat()
            assert [st.st_size,st.st_mtime_ns]==identity,rel
    return cfg


def gate_identity(out):
    out=Path(out);f=read(out/'gate-model-freeze.json')
    assert f['fit_count']==1 and f['training_pairs_sha256']==sha(out/'train-pairs.json')
    assert sha(out/'gate-model.json')==f['gate_sha256'],'Fitted gate identity changed'
    for name in ('fit-result.json','development-gate.json'):
        if (out/name).exists():assert read(out/name)['gate_sha256']==f['gate_sha256']
    return f['gate_sha256']


def ready(out=OUT):
    out=Path(out); cfg=verify(out,history=True)
    review=read(out/'pre-review.json'); receipt=read(out/'pre-execution-commit.json')
    assert review['passed'] and review['freeze_sha256']==sha(out/'freeze.json')
    assert review['registration_sha256']==sha(out/'registration.json')
    for name in ('registration.json','freeze.json','pre-review.json'):
        assert git('show',receipt['commit']+':review/evidence/'+out.name+'/'+name)==(out/name).read_bytes()
    for rel in read(out/'freeze.json')['runtime_files']:
        import hashlib
        published=git('show',receipt['commit']+':kaggriculture/'+rel)
        assert hashlib.sha256(published).hexdigest()==read(out/'freeze.json')['published_runtime_files'][rel]
        assert line_endings_only((ROOT/rel).read_bytes(),published),rel
    return cfg


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','freeze','verify'])
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    cfg=prepare(args.out) if args.command=='prepare' else freeze(args.out) if args.command=='freeze' else verify(args.out,history=True)
    print(json.dumps(dict(experiment=cfg['experiment'], passed=True)))
