"""Immutable registration and stage entry gates for one task imitation fit."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from learning.bc_crop.agent import ROOT, TEACHER_SHA, sha
from learning.bc_crop.experiment import ENGINE_COMMIT, ENGINE_SHA, read, save

PREVIOUS = ROOT / 'artifacts/task-choice-v1-20261023'
CROP = ROOT / 'artifacts/bc-epoch6-holdout-v1-20261023/panels/fixed_epoch6/model/trained.npz'
CROP_SHA = 'd497e8284dde1f7b836b1cf88915822f82a96be1f99e7b126d06ea263e281ffd'


def git(public, *args):
    return subprocess.check_output(['git', '-C', str(public), *args])


def prepare(out, public):
    out, public = Path(out).resolve(), Path(public).resolve()
    if out.exists():
        raise FileExistsError('Preserve existing experiment')
    assert not git(public, 'status', '--porcelain').strip()
    from decision.verify_mechanisms import rules
    assert sha(Path(rules.__file__)) == ENGINE_SHA
    assert sha(CROP) == CROP_SHA
    assert sha(ROOT / 'bots/teacher_fertilizer_terminal_v2.py') == TEACHER_SHA
    advice = read(PREVIOUS / 'next-stage-advice.json')
    assert advice['training']['epochs'] == 8
    task_cfg = read(PREVIOUS / 'registration.json')
    old = read(PREVIOUS / 'protection-before.json')['runtime'].copy()
    for base in (PREVIOUS, ROOT / 'learning/task_choice_v1'):
        for path in base.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                old[path.relative_to(ROOT).as_posix()] = sha(path)
    for rel, digest in old.items():
        assert sha(ROOT / rel) == digest, rel
    tracked = git(public, 'ls-files', '-z').decode('utf-8').split('\0')
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in (
        PREVIOUS / 'dataset-manifest.json', PREVIOUS / 'schema.json', PREVIOUS / 'coverage.json',
        PREVIOUS / 'acceptance.json', PREVIOUS / 'next-stage-advice.json',
        PREVIOUS / 'reconstruction-ledger.json', PREVIOUS / 'registration.json',
        ROOT / task_cfg['demonstration_root'] / 'registration.json',
        ROOT / task_cfg['demonstration_root'] / 'collect-ledger.json',
        ROOT / task_cfg['demonstration_root'] / 'collect-smoke-ledger.json', CROP)}
    cfg = dict(experiment='joint-task-bc-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git(public, 'rev-parse', 'HEAD').decode().strip(), initial_worktree_clean=True,
        engine_commit=ENGINE_COMMIT, engine_sha256=ENGINE_SHA, teacher_sha256=TEACHER_SHA,
        task_dataset_root=PREVIOUS.relative_to(ROOT).as_posix(),
        demonstration_root=task_cfg['demonstration_root'],
        crop_checkpoint=CROP.relative_to(ROOT).as_posix(), crop_checkpoint_sha256=CROP_SHA,
        train_seeds=task_cfg['train_seeds'], validation_seeds=task_cfg['validation_seeds'],
        input_files=inputs, model_seed=2026102331,
        training=advice['training'], model=advice['model'], offline_entry_gate=advice['offline_entry_gate'],
        development_games=advice['development_games'], workers=1,
        extra_demonstrations=0, new_formal_holdout=0, default_changed=False,
        normalization_denominators={'shared': 'train decision groups', 'candidate': 'train real candidate rows'},
        task_candidate_scope='frozen teacher economic opportunities, not all official legal actions',
        data_role='Original8train/2validation only; development, not independent testing',
        cache='All original180files/135406groups; ragged275+98, no truncation/drop/oversampling',
        numerical_upper_bound='Optimistic deterministic complete ordered numeric input consistency bound, not MLP attainable accuracy',
        composition='Four original validation round2-seat observation trajectories; TaskAgent teacher task+epoch6 versus CropAgent epoch6 actions/crop/reservations; not original teacher raw action or new closed-loop equivalence',
        training_seed_scope='single fixed model/batch generator; no search or restart',
        acceptance='Eight epochs complete; frozen offline gates, whole validation independent NumPy export identity, actual action or reservation influence and privacy isolation before any games. No epoch passes means zero games.',
        game_acceptance='All attempted games720states/719actions/DONE both, no errors/fallback; ordered official replay, actor+rival actions, task/crop traces and cash reproduced; maximum12development+2entry repeats.',
        stop='No extension, retries, relaxed gates, replacement games, additional data, PPO or DAgger. Failure consumes budget; old evidence retained.',
        limitations=advice['limitations'])
    # Frozen settings have explicit numeric fields; prose is not execution policy.
    cfg['training'] = dict(cfg['training'], seed=cfg['model_seed'], batch_size=256,
        padded_candidate_budget=65536, normalization_std_floor=1., class_weight_clip=[.5,5.])
    out.mkdir(parents=True)
    (out / 'sources').mkdir()
    sources = {'worker.py': ROOT / 'learning/round2_worker.py',
               'round2.py': ROOT / cfg['demonstration_root'] / 'sources/round2.py'}
    original=read(ROOT/cfg['demonstration_root']/'registration.json')
    assert sha(sources['round2.py'])==original['source_sha256']['round2.py']
    for name, path in sources.items():
        shutil.copyfile(path, out / 'sources' / name)
    cfg['source_sha256'] = {n: sha(p) for n,p in sources.items()}
    save(out / 'registration.json', cfg)
    save(out / 'protection-before.json', {'runtime': old, 'public': {n: sha(public/n) for n in tracked if n}})
    files = [*sorted((ROOT/'learning/task_bc_v1').glob('*.py')),
             ROOT/'learning/task_bc_v1/README.zh-CN.md',
             *sorted((ROOT/'learning/task_choice_v1').glob('*.py')),
             *sorted((ROOT/'learning/bc_crop').glob('*.py')),
             *sorted((ROOT/'decision').glob('*.py')),
             ROOT/'learning/round2_worker.py', ROOT/'learning/optimize_round2.py',
             ROOT/'learning/evaluate_holdout.py', ROOT/'run_match.py', ROOT/'replay_metrics.py',
             ROOT/'localize_replay.py', ROOT/'bots/teacher_fertilizer_terminal_v2.py']
    save(out/'freeze.json', dict(runtime_files={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        registration_sha256=sha(out/'registration.json'), protection_sha256=sha(out/'protection-before.json'),
        training_results_absent=True, local_freeze_not_external_timestamp=True))
    return verify(out)


def verify(out, history=False):
    out = Path(out)
    cfg, freeze = read(out/'registration.json'), read(out/'freeze.json')
    assert sha(out/'registration.json') == freeze['registration_sha256']
    assert sha(out/'protection-before.json') == freeze['protection_sha256']
    for rel, digest in freeze['runtime_files'].items():
        assert sha(ROOT/rel) == digest, rel
    for rel, digest in cfg['input_files'].items():
        assert sha(ROOT/rel) == digest, rel
    for name, digest in cfg['source_sha256'].items():
        assert sha(out/'sources'/name) == digest, name
    if history:
        for rel, digest in read(out/'protection-before.json')['runtime'].items():
            assert sha(ROOT/rel) == digest, rel
    return cfg


def ready(out, phase='prepare_data'):
    out = Path(out)
    cfg = verify(out)
    review, receipt = read(out/'pre-review.json'), read(out/'pre-training-commit.json')
    assert review['passed'] and review['freeze_sha256'] == sha(out/'freeze.json')
    public = ROOT.parent/'publication/ai-game-lab'
    for name in ('registration.json', 'freeze.json', 'pre-review.json'):
        raw = git(public, 'show', receipt['commit']+':review/evidence/'+out.name+'/'+name)
        assert raw == (out/name).read_bytes(), name
    if phase == 'train':
        receipt = read(out/'pre-training-data-review.json')
        assert receipt['passed']
        for name, digest in receipt['evidence_sha256'].items():
            assert sha(out/name) == digest, name
        summary=read(out/'cache-summary.json')
        assert summary['passed'] and summary['all_groups_retained']
        assert summary['source_files_verified']==180
        for split in ('train','validation'):
            assert summary['splits'][split]['max_group_size']<=65536
        assert summary['splits']['validation']['multicandidate_only']['optimistic_exact_id_accuracy_upper_bound']>=.6
        assert read(out/'composition-parity.json')['passed']
    elif phase != 'prepare_data':
        raise ValueError('Unknown experiment phase')
    return cfg


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare','verify','cache','parity'))
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--public', type=Path)
    a=p.parse_args()
    if a.mode == 'prepare': prepare(a.out,a.public)
    elif a.mode == 'verify': verify(a.out,history=True)
    elif a.mode == 'cache':
        from .cache import run
        run(a.out)
    else:
        from .parity import run
        run(a.out)
