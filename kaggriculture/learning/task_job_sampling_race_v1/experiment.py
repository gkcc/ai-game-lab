"""One-shot preregistration gate for the artificial Job sampling scenario."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil
from . import scenario

ROOT=Path(__file__).resolve().parents[2]
PUBLIC=ROOT.parent/'publication/ai-game-lab'
OUT=ROOT/'artifacts/task-job-sampling-race-v1-20261027'
PARENT=ROOT/'artifacts/task-teacher-choice-eval-v1-20261027'
SOURCE=Path(__file__).resolve().parent
JOB_SOURCE=ROOT/'learning/task_training_job_supervision_v1/job.py'
EVIDENCE=('root-result-acceptance.json','registration.json','freeze.json','failure-projection.json','independent-failure-assessment.json')


class ReceiptFailure(RuntimeError):
    def __init__(self,primary,receipt):
        super().__init__('Synthetic receipt failed')
        self.primary_error_kind,self.receipt_error_kind=primary,receipt


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def git(*args): return subprocess.check_output(['git',*args],cwd=PUBLIC)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def write_new(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT): raise ValueError('Output outside experiment')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')


def deadline_check():
    deadline=datetime.fromisoformat(read(OUT/'root-decision.json')['deadline_utc'])
    if deadline.tzinfo is None or datetime.now(timezone.utc)>=deadline: raise TimeoutError('Invalid or expired deadline')


def clean_head(expected):
    if git('rev-parse','HEAD').decode().strip()!=expected or git('status','--porcelain').strip():
        raise ValueError('Unexpected or dirty public HEAD')


def environment():
    venv=ROOT/'.venv'; executable=venv/'Scripts/python.exe'
    try:
        if (not os.path.samefile(sys.executable,executable) or not os.path.samefile(sys.prefix,venv)
            or os.path.samefile(sys.prefix,sys.base_prefix)): raise ValueError('Project virtual environment required')
    except OSError as exc: raise ValueError('Virtual environment identity unavailable') from exc
    return dict(python_version=list(sys.version_info[:3]),psutil_version=psutil.__version__,platform=sys.platform,
                interpreter_relative='.venv/Scripts/python.exe',interpreter_sha256=sha(executable))


def validate_parent(result):
    if (result.get('passed') is not False or result.get('panel_completed') is not False
        or result.get('no_economic_comparison') is not True or result.get('no_retries') is not True):
        raise ValueError('Expected failed unfinished panel evidence missing')


def prepare(out=OUT):
    out=Path(out).resolve()
    if os.name!='nt': raise ValueError('Registration requires Windows')
    if out!=OUT or any((out/n).exists() for n in ('registration.json','freeze.json','started.json')):
        raise ValueError('Registration or attempt already exists')
    deadline_check()
    decision=read(out/'root-decision.json')
    baseline=decision['baseline_head']
    if (decision.get('status')!='approved_for_implementation_not_execution' or decision.get('synthetic_only') is not True
        or decision.get('limits') != scenario.LIMITS or not isinstance(baseline,str) or len(baseline)!=40
        or any(x not in '0123456789abcdef' for x in baseline)):
        raise ValueError('Invalid prospective decision')
    clean_head(baseline)
    for name in EVIDENCE:
        if git('show',baseline+':review/evidence/'+PARENT.name+'/'+name)!=(PARENT/name).read_bytes():
            raise ValueError('Published parent evidence changed')
    result=read(PARENT/'root-result-acceptance.json'); validate_parent(result)
    for name in EVIDENCE[1:]:
        if sha(PARENT/name)!=result['evidence_sha256'][name]: raise ValueError('Parent evidence chain changed')
    if sha(PARENT/'registration.json')!=read(PARENT/'freeze.json')['registration_sha256']:
        raise ValueError('Parent registration changed')
    if git('show',baseline+':kaggriculture/'+JOB_SOURCE.relative_to(ROOT).as_posix()) != JOB_SOURCE.read_bytes():
        raise ValueError('Published Job source changed')
    source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(SOURCE.glob('*.py'))}
    cfg=dict(experiment='task-job-sampling-race-v1',baseline_head=baseline,deadline_utc=decision['deadline_utc'],
        synthetic_only=True,limits=dict(scenario.LIMITS),parent_evidence={n:sha(PARENT/n) for n in EVIDENCE},
        job_source_sha256=sha(JOB_SOURCE),environment=environment(),
        workload=dict(stable_mib=24,stable_seconds=8,max_short_children=16,short_seconds=0.05,artificial_injections=1),
        boundary='No game, model, training cache or replay access; artificial injection is not evidence of prior cause',
        attempts=1,retry_allowed=False)
    write_new(out/'registration.json',cfg)
    write_new(out/'freeze.json',dict(registration_sha256=sha(out/'registration.json'),root_decision_sha256=sha(out/'root-decision.json'),
        new_source_sha256=source,shared_source_sha256={JOB_SOURCE.relative_to(ROOT).as_posix():cfg['job_source_sha256']},baseline_head=baseline))
    return verify(out)


def verify(out=OUT):
    out=Path(out).resolve()
    if out!=OUT: raise ValueError('Unexpected output')
    deadline_check()
    cfg,freeze=read(out/'registration.json'),read(out/'freeze.json')
    receipt=out/'pre-execution-commit.json'
    clean_head(read(receipt)['commit'] if receipt.exists() else cfg['baseline_head'])
    if sha(out/'registration.json')!=freeze['registration_sha256'] or sha(out/'root-decision.json')!=freeze['root_decision_sha256']:
        raise ValueError('Registration changed')
    if cfg['limits']!=scenario.LIMITS or cfg['environment']!=environment() or cfg['synthetic_only'] is not True:
        raise ValueError('Scenario or environment changed')
    if {p.relative_to(ROOT).as_posix() for p in SOURCE.glob('*.py')} != set(freeze['new_source_sha256']): raise ValueError('Source set changed')
    for rel,digest in freeze['new_source_sha256'].items():
        if sha(ROOT/rel)!=digest: raise ValueError('Source changed')
    job_digest=freeze['shared_source_sha256'][JOB_SOURCE.relative_to(ROOT).as_posix()]
    if sha(JOB_SOURCE)!=job_digest or cfg['job_source_sha256']!=job_digest:
        raise ValueError('Job source changed')
    for name,digest in cfg['parent_evidence'].items():
        if sha(PARENT/name)!=digest: raise ValueError('Parent evidence changed')
    return cfg


def verify_commit(out=OUT):
    out=Path(out).resolve(); cfg=verify(out); receipt=read(out/'pre-execution-commit.json'); commit=receipt['commit']
    clean_head(commit)
    if git('merge-base','--is-ancestor',cfg['baseline_head'],commit)!=b'': raise ValueError('Invalid commit ancestry')
    for name in ('root-decision.json','registration.json','freeze.json','pre-review.json'):
        if git('show',commit+':review/evidence/'+out.name+'/'+name)!=(out/name).read_bytes(): raise ValueError('Committed evidence changed')
    for rel in read(out/'freeze.json')['new_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel)!=(ROOT/rel).read_bytes(): raise ValueError('Committed source changed')
    for rel in read(out/'freeze.json')['shared_source_sha256']:
        if git('show',commit+':kaggriculture/'+rel)!=(ROOT/rel).read_bytes(): raise ValueError('Committed shared source changed')
    return cfg


def ready(out=OUT):
    cfg=verify(out); review=read(Path(out)/'pre-review.json'); receipt=read(Path(out)/'pre-execution-commit.json')
    if (review.get('passed') is not True or review.get('freeze_sha256')!=sha(Path(out)/'freeze.json')
        or receipt.get('freeze_sha256')!=sha(Path(out)/'freeze.json')
        or receipt.get('registration_sha256')!=sha(Path(out)/'registration.json')):
        raise ValueError('Review or commitment changed')
    verify_commit(out)
    return cfg


def run(out=OUT,*,api_factory=None,execute=None):
    from learning.task_training_job_supervision_v1.job import WindowsAPI
    out=Path(out).resolve(); ready(out)
    if (out/'started.json').exists() or (out/'result.json').exists(): raise ValueError('Attempt already consumed')
    marker=dict(created=False,complete=False); result=None; error=None
    identities=dict(marker_sha256=None,freeze_sha256=None,registration_sha256=None,
                    pre_execution_commit_sha256=None,commit=None)
    try:
        path=(out/'started.json').resolve()
        if not path.is_relative_to(OUT): raise ValueError('Marker outside experiment')
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            marker['created']=True
            json.dump(dict(attempt=1,freeze_sha256=sha(out/'freeze.json'),retry_allowed=False),stream); stream.write('\n')
        marker['complete']=True
        identities.update(marker_sha256=sha(out/'started.json'),freeze_sha256=sha(out/'freeze.json'),
            registration_sha256=sha(out/'registration.json'),pre_execution_commit_sha256=sha(out/'pre-execution-commit.json'),
            commit=read(out/'pre-execution-commit.json')['commit'])
        result=(execute or scenario.run_owned)((api_factory or WindowsAPI)(),sys.executable,ROOT)
        if (result.get('passed') is not True or not scenario.accepted(result.get('supervision',{}),
                                                                       result.get('artificial_disappearances_injected'))):
            raise RuntimeError('Synthetic scenario not accepted')
        ready(out); verify_commit(out); deadline_check()
        if (sha(out/'started.json') != identities['marker_sha256'] or sha(out/'freeze.json') != identities['freeze_sha256']
            or sha(out/'registration.json') != identities['registration_sha256']
            or sha(out/'pre-execution-commit.json') != identities['pre_execution_commit_sha256']
            or read(out/'pre-execution-commit.json')['commit'] != identities['commit']):
            raise ValueError('Execution identity changed')
    except BaseException as exc: error=type(exc).__name__
    receipt=dict(passed=error is None,marker=marker,error_kind=error,result=result,identities=identities,retry_allowed=False,
        synthetic_only=True,artificial_injection_not_prior_cause=True,
        identity_check_limit='Before and after checks cannot exclude transient replacement followed by restoration')
    if marker['created']:
        try: write_new(out/'result.json',receipt)
        except BaseException as exc: raise ReceiptFailure(error,type(exc).__name__) from None
    if error: raise RuntimeError('Synthetic attempt failed: '+error)
    return receipt


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('prepare','verify','ready','verify_commit','run'))
    args=parser.parse_args(); result=globals()[args.command]()
    print(json.dumps(dict(passed=result.get('passed',True))))
