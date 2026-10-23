"""Freeze and verify a zero-training task-interface reconstruction stage."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from learning.bc_crop.agent import ROOT, TEACHER_SHA, sha
from learning.bc_crop.experiment import ENGINE_COMMIT, ENGINE_SHA, read, save

PREVIOUS = ROOT/'artifacts/bc-epoch6-holdout-v1-20261023'
DEMONSTRATIONS = ROOT/'artifacts/bc-crop-v1-20261023-run1'


def git(public, *args):
    return subprocess.check_output(['git','-C',str(public),*args], text=True).strip()


def prepare(out, public):
    out, public = out.resolve(), public.resolve()
    if out.exists():raise FileExistsError('Preserve existing experiment')
    assert not git(public,'status','--porcelain'), 'Record a clean baseline before preparation'
    previous = read(PREVIOUS/'acceptance.json')
    assert previous['research_baseline_admitted'] and not previous['default_changed']
    original=read(DEMONSTRATIONS/'registration.json')
    assert sha(ROOT/'bots/teacher_fertilizer_terminal_v2.py') == TEACHER_SHA
    from decision.verify_mechanisms import rules
    assert sha(Path(rules.__file__)) == ENGINE_SHA
    tracked=[p for p in subprocess.check_output(['git','-C',str(public),'ls-files','-z']).decode('utf-8').split('\0') if p]
    protected_public={p:sha(public/p) for p in tracked}
    old=read(PREVIOUS/'protection-before.json')['runtime']
    protected_runtime={str(Path(p).relative_to('kaggriculture')).replace('\\','/'):h for p,h in old.items()}
    for p in PREVIOUS.rglob('*'):
        if p.is_file():protected_runtime[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in [ROOT/'main.py',*sorted((ROOT/'bots').glob('*.py')),
              *sorted((ROOT/'models').rglob('*.npz')),*sorted((ROOT/'models').rglob('*.json'))]:
        protected_runtime[p.relative_to(ROOT).as_posix()]=sha(p)
    for rel,h in protected_runtime.items():assert sha(ROOT/rel)==h,rel
    files=[*sorted((ROOT/'learning/task_choice_v1').glob('*.py')),
           ROOT/'learning/task_choice_v1/README.zh-CN.md',
           *sorted((ROOT/'learning/bc_crop').glob('*.py')),
           *sorted((ROOT/'decision').glob('*.py')),ROOT/'bots/teacher_fertilizer_terminal_v2.py']
    cfg=dict(experiment='joint-task-choice-interface-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_head=git(public,'rev-parse','HEAD'),initial_worktree_clean=True,
        engine_commit=ENGINE_COMMIT,engine_sha256=ENGINE_SHA,teacher_sha256=TEACHER_SHA,
        demonstration_root=DEMONSTRATIONS.relative_to(ROOT).as_posix(),
        ledger_names=['collect-ledger.json','collect-smoke-ledger.json'],
        source_ledgers={n:sha(DEMONSTRATIONS/n) for n in ('collect-ledger.json','collect-smoke-ledger.json')},
        train_seeds=original['train_seeds'],validation_seeds=original['validation_seeds'],
        workers=4,demonstrations=60,training_runs=0,new_complete_games=0,
        new_official_step_pairs=6,max_new_official_transitions=12,
        existing_replay_transitions=43140,
        selected_task_types=['PLANT','DIG','WATER','HARVEST','FERTILIZE'],
        candidate='joint unit + target + operation; canonical order independent of teacher scores',
        input='sanitized deployable view plus actual-path reservations; no teacher priority/score/rank/label or offline identities',
        labels='separate teacher-selected candidate ID/index; original crop dataset is not reused as task labels',
        protocol_scope='teacher economic opportunity generator and DROP/PICKUP/navigation/market remain rules; not a full legal mask',
        constructed_cases=['PLANT','DIG','WATER','HARVEST','FERTILIZE','joint_PLANT'],
        case_selection='Six fixed constructed boundary states, not searched replay cases. Code frozen before official execution; one baseline/alternative step each.',
        fixture_preflight='Observation-only preflight adjusted draft HARVEST to day28/planted20/stock9600 so original teacher has both fertilizer and harvest opportunities. Zero official transitions before freeze.',
        acceptance=['all60complete actor actions match raw replay and frozen teacher, rival actions match, official replay v1/v2/final rewards match',
                    'five task types selected in both original train/validation splits',
                    'each type has selected examples with competing task types in both splits',
                    'callback has direct official control over all five operations and actual second-unit reservations',
                    'input isolation, invalid-selection rejection, cross-episode reset, source identity and history/privacy checks pass'],
        stop='Any failed case/reconstruction/coverage is retained; no automatic retries, extra demos, relaxed gate, new games, training or PPO/DAgger',
        private_holdout_steps_used=False,default_changed=False,
        prior_baseline_model_sha256=previous['checkpoint_sha256'])
    from .dataset import selected_rows
    selected_rows(cfg)
    out.mkdir(parents=True)
    save(out/'registration.json',cfg)
    save(out/'protection-before.json',{'runtime':protected_runtime,'public':protected_public})
    runtime={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    for p in (ROOT/'learning/task_choice_v1').iterdir():
        if p.is_file():
            target=out/'sources'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
    save(out/'freeze.json',{'runtime_files':runtime,'registration_sha256':sha(out/'registration.json'),
                          'data_and_official_case_results_absent':True,'local_freeze_not_external_timestamp':True})
    verify(out)
    return cfg


def verify(out):
    out=Path(out);cfg=read(out/'registration.json');freeze=read(out/'freeze.json')
    assert sha(out/'registration.json')==freeze['registration_sha256']
    for rel,h in freeze['runtime_files'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in read(out/'protection-before.json')['runtime'].items():assert sha(ROOT/rel)==h,rel
    for n,h in cfg['source_ledgers'].items():assert sha(ROOT/cfg['demonstration_root']/n)==h
    assert cfg['training_runs']==cfg['new_complete_games']==0 and cfg['demonstrations']==60
    assert cfg['max_new_official_transitions']==12
    from decision.verify_mechanisms import rules
    assert sha(Path(rules.__file__))==cfg['engine_sha256']==ENGINE_SHA
    return cfg


def ready(out):
    out=Path(out);cfg=verify(out)
    review=read(out/'pre-review.json');receipt=read(out/'pre-game-commit.json')
    assert review['passed'] and review['freeze_sha256']==sha(out/'freeze.json')
    public=ROOT.parent/'publication/ai-game-lab'
    for n in ('registration.json','freeze.json','pre-review.json'):
        raw=subprocess.check_output(['git','-C',str(public),'show',receipt['commit']+':review/evidence/'+out.name+'/'+n])
        assert raw==(out/n).read_bytes(),n
    return cfg


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','verify','cases','reconstruct'))
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--public',type=Path)
    a=parser.parse_args()
    if a.mode=='prepare':prepare(a.out,a.public)
    elif a.mode=='verify':verify(a.out)
    elif a.mode=='cases':
        from .cases import run
        run(a.out)
    else:
        from .dataset import run
        run(a.out)
