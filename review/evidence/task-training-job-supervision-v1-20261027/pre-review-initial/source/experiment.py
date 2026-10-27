"""Prerequisite freeze for two synthetic-only owned Windows Job scenarios."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil
from .scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT.parent/'publication/ai-game-lab'
OUT = ROOT/'artifacts/task-training-job-supervision-v1-20261027'
PARENT = ROOT/'artifacts/task-completion-training-coverage-v1-20261027'
SOURCE = Path(__file__).resolve().parent
BASE = 'd7c9782591ba884d38124f1910ad8f0650b7d913'
EVIDENCE = ('root-result-acceptance.json','registration.json','freeze.json','supervisor-result.json',
            'launcher-counterexample.json','process-search-correction.json','independent-result-assessment.json',
            'root-next-learning-decision.json')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def write_new(path,value):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT): raise ValueError('Output outside experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def deadline_check():
    deadline = datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline: raise TimeoutError('Invalid or expired deadline')


def clean_head(expected):
    if git('rev-parse','HEAD').decode().strip() != expected or git('status','--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def validate_parent(result,counterexample,correction):
    if (result.get('passed') is not False or result.get('resource_supervision_accepted') is not False
        or result.get('resource_peak_of_worker_unknown') is not True
        or counterexample.get('synthetic_runs') != 1 or counterexample.get('descendant_process_count',0) < 1
        or counterexample.get('original_analysis_repeated') is not False
        or correction.get('inaccessible_processes_present') is not True
        or correction.get('real_analysis_repeated') is not False):
        raise ValueError('Expected negative parent evidence missing')


def environment():
    venv = ROOT/'.venv'
    executable = venv/'Scripts/python.exe'
    try:
        if (not os.path.samefile(sys.executable,executable) or not os.path.samefile(sys.prefix,venv)
                or os.path.samefile(sys.prefix,sys.base_prefix)):
            raise ValueError('Project virtual environment required')
    except OSError as exc:
        raise ValueError('Project virtual environment identity unavailable') from exc
    return dict(python_version=list(sys.version_info[:3]),psutil_version=psutil.__version__,platform=sys.platform,
                interpreter_relative='.venv/Scripts/python.exe',interpreter_sha256=sha(executable),
                prefix_relative='.venv',virtual_environment_verified=True)


def prepare(out=OUT):
    out = Path(out).resolve()
    if os.name != 'nt': raise ValueError('Registration requires Windows')
    if out != OUT or any((out/name).exists() for name in ('registration.json','freeze.json','normal-started.json','rss_stop-started.json')):
        raise ValueError('Registration or scenario already exists')
    deadline_check()
    decision = read(out/'root-decision.json')
    if decision.get('status') != 'approved_for_implementation_not_execution' or decision.get('previous_commit') != BASE:
        raise ValueError('Invalid root decision')
    clean_head(BASE)
    for name in EVIDENCE:
        if git('show',BASE+':review/evidence/'+PARENT.name+'/'+name) != (PARENT/name).read_bytes():
            raise ValueError('Published parent evidence changed')
    parent = read(PARENT/'root-result-acceptance.json')
    validate_parent(parent,read(PARENT/'launcher-counterexample.json'),read(PARENT/'process-search-correction.json'))
    for name in EVIDENCE[1:]:
        if name in parent['evidence_sha256'] and sha(PARENT/name) != parent['evidence_sha256'][name]:
            raise ValueError('Parent evidence chain changed')
    freeze = read(PARENT/'freeze.json')
    if sha(PARENT/'registration.json') != freeze['registration_sha256']: raise ValueError('Parent registration changed')
    source = {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    cfg = dict(experiment='task-training-job-supervision-v1',baseline_head=BASE,deadline_utc=decision['deadline_utc'],
        parent_evidence={n:sha(PARENT/n) for n in EVIDENCE},scenarios=SCENARIOS,environment=environment(),
        synthetic_only=True,cache_reads=0,model_reads=0,replay_reads=0,new_games=0,training_updates=0,
        ownership='Only newly created Job and process handles; no other processes',
        sampling_limit='Job membership and RSS are sampled, may race, and are not hard real-time limits',
        member_interpretation='Two sampled members do not independently identify the worker; normal launcher zero exit implies its preset worker exited successfully',
        retries=False)
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),root_decision_sha256=sha(out/'root-decision.json'),
        new_source_sha256=source,parent_commit=BASE))
    return verify(out)


def verify(out=OUT):
    out = Path(out).resolve()
    if out != OUT: raise ValueError('Unexpected output')
    deadline_check()
    receipt = out/'pre-execution-commit.json'
    clean_head(read(receipt)['commit'] if receipt.exists() else BASE)
    cfg,freeze = read(out/'registration.json'),read(out/'freeze.json')
    if sha(out/'registration.json') != freeze['registration_sha256'] or sha(out/'root-decision.json') != freeze['root_decision_sha256']:
        raise ValueError('Registration changed')
    if cfg['scenarios'] != SCENARIOS or cfg['environment'] != environment(): raise ValueError('Scenarios or environment changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} != set(freeze['new_source_sha256']): raise ValueError('Source set changed')
    for rel,digest in freeze['new_source_sha256'].items():
        if sha(ROOT/rel) != digest: raise ValueError('Source changed')
    for name,digest in cfg['parent_evidence'].items():
        if sha(PARENT/name) != digest: raise ValueError('Parent evidence changed')
    return cfg


def ready(out=OUT):
    cfg = verify(out)
    review,receipt = read(Path(out)/'pre-review.json'),read(Path(out)/'pre-execution-commit.json')
    if (not review['passed'] or review['freeze_sha256'] != sha(Path(out)/'freeze.json')
        or receipt['freeze_sha256'] != sha(Path(out)/'freeze.json') or receipt['registration_sha256'] != sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    verify_commit(out)
    return cfg


def verify_commit(out=OUT):
    out = Path(out).resolve()
    cfg = verify(out)
    commit = read(out/'pre-execution-commit.json')['commit']
    clean_head(commit)
    if git('merge-base','--is-ancestor',BASE,commit) != b'': raise ValueError('Invalid ancestry')
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name) != (out/name).read_bytes(): raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel) != (ROOT/rel).read_bytes(): raise ValueError('Committed source changed')
    return cfg
