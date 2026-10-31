"""Single preregistered in-sample diagnostic of a frozen state residual.

No games, training or GPU work. Only the already-used 42 training groups are
scored. Historical diagnostic arrays may be opened by their existing seal
verifier but are never scored or summarized here. Inputs and models stay
private; importing this module performs no source access.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time

from learning.task_student_continuation_v1 import experiment as previous
from learning.task_multi_teacher_probe_v1 import experiment as supervisor
from learning.task_teacher_transfer_dataset_v1 import experiment as transfer
from learning.task_teacher_cross_source_v1 import experiment as cross
from learning.task_disagreement_trace_v1 import experiment as source
from learning.task_state_residual_v1 import experiment as state
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA


REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-state-train-fit-v1-20261031'
REGISTRATION = REPO / 'review/evidence/task-state-train-fit-v1-20261031/registration.json'
BASE_COMMIT = 'e14c037d213b197427fcdec39c53f723a3727e28'
STATE_COMMIT = '1d3f9ff3ebe4b1dfb1bf648e14f05b0cc50e19bd'
SOURCE_COMMIT = '384bc5f2605b157e160ba3b1910151d2eeaf53ee'
JOB = dict(supervisor.JOB)
INITIAL_MEMORY, MIN_DISK, OUTPUT_RESERVE = 5 * 1024**3, 4 * 1024**3, 4 * 1024**2
SOURCE_NAMES = tuple(dict.fromkeys(state.SOURCE_NAMES + (
    'kaggriculture/learning/task_state_train_fit_v1/__init__.py',
    'kaggriculture/learning/task_state_train_fit_v1/analysis.py',
    'kaggriculture/learning/task_state_train_fit_v1/experiment.py',
    'kaggriculture/learning/task_state_train_fit_v1/test_analysis.py',
    'kaggriculture/learning/task_state_train_fit_v1/test_experiment.py',
    'review/evidence/task-state-train-fit-v1-20261031/REGISTRATION.zh-CN.md',
)))


def _git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],
        stderr=subprocess.DEVNULL, timeout=30).decode('utf-8').strip()


def _read(path, limit=1024**2):
    return cross._read(path, limit=limit)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_hashes():
    return {name:_sha(_read(REPO / name)) for name in SOURCE_NAMES}


def _registration(source_sha, binding_sha):
    return dict(schema='state-train-fit-prereg-v1', base_commit=BASE_COMMIT,
        state_registration_commit=STATE_COMMIT, disagreement_registration_commit=SOURCE_COMMIT,
        source_sha256=source_sha, private_binding_sha256=binding_sha,
        official_rule_commit=ENGINE_COMMIT, official_rule_sha256=ENGINE_SHA,
        train_groups=42, role_labels=['A03','A04'], actual_fit_pool_proof=True,
        baseline_student_index_required=True, score_scope='frozen_train_groups_only',
        statistics=['teacher_match','choice_change','target_nll','teacher_vs_best_other_margin'],
        margin_delta_abs_tolerance=1e-6,
        margin_classification='above_tolerance_below_negative_tolerance_otherwise_within_tolerance',
        diagnostic_scoring=False, training=False, model_updates=0, new_games=0,
        gpu=False, device='cpu', independent_evaluation=False, promotion=False,
        job=dict(JOB), initial_memory_bytes=INITIAL_MEMORY, disk_floor_bytes=MIN_DISK,
        output_reserve_bytes=OUTPUT_RESERVE, max_attempts=1)


def _all_train_in_fit(result):
    """Validate aggregate proof of membership in fit's input pool, not draws."""
    if (type(result) is not dict or result.get('registration_commit') != STATE_COMMIT
            or result.get('train_student_checked') != dict(groups=42,mismatches=0)
            or result.get('screen',{}).get('screen') != 'failed'):
        raise ValueError('Frozen training result differs')
    coverage = result.get('coverage')
    state._validate_coverage(coverage)
    split, filtered = coverage['split'], coverage['filtered']['new_train']
    if (coverage['passed'] is not True or split['input_by_pool']['new_train'] != 42
            or split['retained_by_pool']['new_train'] != 42
            or any(split[key]['new_train'] != 0 for key in ('duplicate_removed_by_pool',
                'conflict_removed_by_pool','diagnostic_excluded_train_connected_by_pool'))
            or filtered != dict(single=0,identical_rows=0,student_same=0)
            or coverage['eligible_by_pool']['new_train'] != 42):
        raise ValueError('Frozen training membership differs')
    return True


def _source_contracts(guard):
    """Authenticate completed seals without scoring or loading a game."""
    guard()
    prior = state.load_completed(guard=guard)
    _all_train_in_fit(prior)
    state_binding, state_sha = state._registered_binding(STATE_COMMIT)
    consumed = source.load_completed(guard=guard)
    source_binding, source_sha = source._registered_binding(SOURCE_COMMIT)
    if (consumed['registration_commit'] != SOURCE_COMMIT or consumed['summary']['data_gate'] != 'passed'
            or consumed['train_sha256'] != state_binding['train_sha256']
            or source_sha != state_binding['new_binding_sha256']
            or prior['baseline_sha256'] != state_binding['baseline_sha256']
            or source_binding['baseline_sha256'] != state_binding['baseline_sha256']
            or source_binding['schema_sha256'] != state_binding['schema_sha256']):
        raise ValueError('Frozen source identities differ')
    identity = dict(state_binding_sha256=state_sha, source_binding_sha256=source_sha,
        baseline_sha256=prior['baseline_sha256'], params_sha256=prior['params_sha256'],
        train_sha256=consumed['train_sha256'], schema_sha256=state_binding['schema_sha256'],
        metadata_sha256=state_binding['metadata_sha256'])
    guard()
    return identity


def prepare():
    if (os.name != 'nt' or _git('rev-parse','HEAD') != BASE_COMMIT
            or _git('diff','--name-only') or _git('diff','--cached','--name-only')
            or PRIVATE.exists() or REGISTRATION.exists()):
        raise ValueError('Unexpected registration state')
    if previous.physical_available() < INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    deadline=time.monotonic()+300
    identity=_source_contracts(lambda:previous._guard(deadline))
    binding=dict(schema='state-train-fit-private-binding-v1',salt=secrets.token_hex(32),**identity)
    document=_registration(_source_hashes(),_sha(previous._encoded(binding)))
    PRIVATE.mkdir(parents=False,exist_ok=False)
    previous._write_new(PRIVATE/'binding.local.json',binding)
    REGISTRATION.parent.mkdir(parents=True,exist_ok=True)
    previous._write_new(REGISTRATION,document)
    print(json.dumps(dict(status='prepared')),flush=True)


def _registered_binding(head):
    if type(head) is not str or re.fullmatch(r'[0-9a-f]{40}',head) is None:
        raise ValueError('Invalid registration commit')
    parent=_git('rev-parse',head+'^')
    raw=subprocess.check_output(['git','-C',str(REPO),'show',
        head+':'+REGISTRATION.relative_to(REPO).as_posix()],stderr=subprocess.DEVNULL,timeout=30)
    binding_raw=_read(PRIVATE/'binding.local.json')
    if parent != BASE_COMMIT or previous._document(raw) != _registration(_source_hashes(),_sha(binding_raw)):
        raise ValueError('Registration differs')
    binding=previous._document(binding_raw)
    keys={'schema','salt','state_binding_sha256','source_binding_sha256','baseline_sha256',
          'params_sha256','train_sha256','schema_sha256','metadata_sha256'}
    if (set(binding)!=keys or binding.get('schema')!='state-train-fit-private-binding-v1'
            or any(type(binding[key]) is not str or re.fullmatch(r'[0-9a-f]{64}',binding[key]) is None
                   for key in keys-{'schema'})):
        raise ValueError('Private source binding differs')
    return binding,_sha(binding_raw)


def _runtime_binding():
    head=_git('rev-parse','HEAD')
    if _git('status','--porcelain') or _git('rev-parse','origin/main') != head:
        raise ValueError('Registration is not clean and published')
    binding,digest=_registered_binding(head)
    return head,binding,digest


def _train_arrays(binding,guard):
    """Read only the separately sealed training NPZ after source validation."""
    guard()
    raw=_read(source.PRIVATE/'train.local.npz',limit=source.MAX_ARCHIVE)
    if _sha(raw) != binding['train_sha256']:
        raise ValueError('Frozen training archive differs')
    arrays=source._load_split_bytes(raw,'train',guard)
    if len(arrays['teacher_target']) != 42:
        raise ValueError('Frozen training population differs')
    return arrays


def _models(binding,guard):
    """Read and authenticate old snapshots without writing their directory."""
    import numpy as np
    from learning.task_bc_v1.model import TaskNetwork
    from learning.task_pair_checkpoint_v1 import snapshot
    guard()
    base_path=state.PRIVATE/'baseline.local.npz'
    base_raw=_read(base_path,limit=snapshot.MAX_ARCHIVE_BYTES)
    decoded=snapshot.load_checkpoint(base_raw,expected_sha256=binding['baseline_sha256'],
                                     expected_schema_sha256=binding['schema_sha256'])
    network=TaskNetwork(base_path,binding['baseline_sha256'])
    if (_sha(previous._encoded(network.metadata)) != binding['metadata_sha256']
            or not np.array_equal(network.mean,decoded.mean) or not np.array_equal(network.scale,decoded.scale)
            or any(not np.array_equal(network.params[key],decoded.params[key]) for key in decoded.params)):
        raise ValueError('Frozen baseline semantics differ')
    params_raw=_read(state.PRIVATE/'params.local.npz',limit=state.MAX_PARAMS)
    if _sha(params_raw) != binding['params_sha256']:
        raise ValueError('Frozen residual differs')
    params=state._decode_params(params_raw)
    guard()
    return network,params


def _score_train(arrays,network,guard):
    """Build exactly 42 groups, checking the recorded student on each."""
    import numpy as np
    from learning.task_multisource_rank_v1 import core as rank
    groups,students,roles=[],[],[]
    if len(arrays['teacher_target']) != 42:
        raise ValueError('Training population differs')
    for i in range(42):
        guard()
        low,high=int(arrays['offsets'][i]),int(arrays['offsets'][i+1])
        shared=np.frombuffer(arrays['shared'][i].tobytes(),dtype=np.float32)
        candidates=np.frombuffer(arrays['candidates'][low:high].tobytes(),dtype=np.float32).reshape(high-low,98)
        student=int(arrays['student_index'][i]); target=int(arrays['teacher_target'][i])
        role_index=int(arrays['role'][i])
        if role_index not in (0,1) or student == target:
            raise ValueError('Training role or recorded disagreement differs')
        scores=network.score(shared,candidates)
        if (type(scores) is not np.ndarray or scores.dtype != np.dtype('float64')
                or scores.shape != (len(candidates),) or not np.isfinite(scores).all()
                or int(np.argmax(scores)) != student):
            raise ValueError('Recorded student choice differs')
        scores=np.frombuffer(scores.tobytes(),dtype=np.float64)
        groups.append(rank.Group(shared,candidates,scores,target,'new'))
        students.append(student);roles.append(source.ROLES[role_index])
    if {role:roles.count(role) for role in ('A03','A04')} != {'A03':21,'A04':21}:
        raise ValueError('Training role population differs')
    guard()
    return groups,students,roles


def _result(raw,head):
    from . import analysis
    if type(analysis.MARGIN_TOLERANCE) is not float or analysis.MARGIN_TOLERANCE != 1e-6:
        raise ValueError('Margin tolerance differs')
    value=previous._document(raw)
    fields={'schema','registration_commit','source_groups','fit_input_membership','student_mismatches',
            'metrics','baseline_sha256','params_sha256','train_sha256','device','training',
            'model_updates','new_games','gpu','independent_evaluation','promotion','software'}
    if (set(value)!=fields or value.get('schema')!='state-train-fit-result-v1'
            or value.get('registration_commit')!=head or type(value.get('source_groups')) is not int
            or value['source_groups']!=42 or value.get('fit_input_membership') is not True
            or type(value.get('student_mismatches')) is not int or value['student_mismatches']!=0
            or value.get('device')!='cpu' or value.get('training') is not False
            or type(value.get('model_updates')) is not int or value['model_updates']!=0
            or type(value.get('new_games')) is not int or value['new_games']!=0
            or value.get('gpu') is not False or value.get('independent_evaluation') is not False
            or value.get('promotion') is not False
            or any(type(value.get(key)) is not str or re.fullmatch(r'[0-9a-f]{64}',value[key]) is None
                   for key in ('baseline_sha256','params_sha256','train_sha256'))
            or type(value.get('software')) is not dict or set(value['software']) != {'python','numpy'}
            or any(type(v) is not str or re.fullmatch(r'[A-Za-z0-9.+_-]{1,64}',v) is None
                   for v in value['software'].values())):
        raise ValueError('Private diagnostic result differs')
    metrics=value['metrics']
    analysis.validate(metrics)
    if (metrics['overall']['count'] != 42 or metrics['overall']['baseline_correct'] != 0
            or set(metrics['by_role']) != {'A03','A04'}
            or any(metrics['by_role'][role]['count'] != 21 for role in ('A03','A04'))):
        raise ValueError('Private diagnostic population differs')
    return value


def public_projection(result):
    """Whitelist only anonymous aggregates; no source/model fingerprints."""
    if type(result) is not dict or type(result.get('registration_commit')) is not str:
        raise ValueError('Invalid diagnostic projection')
    result=_result(previous._encoded(result),result['registration_commit'])
    return dict(schema='state-train-fit-public-v1',registration_commit=result['registration_commit'],
        state_registration_commit=STATE_COMMIT,source_groups=42,fit_input_membership=True,
        student_mismatches=0,metrics=result['metrics'],device='cpu',training=False,
        model_updates=0,new_games=0,gpu=False,independent_evaluation=False,promotion=False)


def child():
    head,binding,binding_sha=_runtime_binding()
    claim=previous._document(_read(PRIVATE/'claim.local.json'))
    attempt=previous._document(_read(PRIVATE/'attempt.local.json'))
    if (claim.get('schema')!='state-train-fit-claim-v1' or claim.get('commit')!=head
            or type(claim.get('attempt')) is not int or claim['attempt']!=1
            or claim.get('freeze_sha256')!=binding_sha or claim.get('retry_allowed') is not False
            or attempt!=dict(schema='state-train-fit-attempt-v1',commit=head,retry_allowed=False)
            or (PRIVATE/'failed.local.json').exists() or (PRIVATE/'closed.local.json').exists()):
        raise ValueError('Child is not uniquely claimed')
    from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY,verify_worker
    from learning.task_pair_fresh_multi_v1.entry import verify_live_owner
    import psutil
    owner=previous._document(_read(PRIVATE/'owner.local.json'))
    verify_worker(os.environ.get(ENV_KEY),claim,binding_sha)
    verify_live_owner(owner,commit=head,binding_sha=binding_sha,current_pid=os.getpid(),
                      process=psutil.Process,executable=sys.executable,samefile=os.path.samefile,
                      now=time.monotonic())
    previous._write_new(PRIVATE/'child-attempt.local.json',dict(
        schema='state-train-fit-child-attempt-v1',commit=head,retry_allowed=False))
    deadline=time.monotonic()+JOB['wall_seconds']
    guard=lambda:previous._guard(deadline)
    phase='source'
    try:
        import numpy as np
        from . import analysis
        identity=_source_contracts(guard)
        if identity != {k:v for k,v in binding.items() if k not in ('schema','salt')}:
            raise ValueError('Source binding changed')
        arrays=_train_arrays(binding,guard)
        network,params=_models(binding,guard)
        phase='train_scoring'
        groups,students,roles=_score_train(arrays,network,guard)
        phase='train_analysis'
        metrics=analysis.assess(groups,students,roles,mean=network.mean,scale=network.scale,
                                params=params,metadata=network.metadata,guard=guard)
        analysis.validate(metrics)
        if (_source_contracts(guard)!=identity or _runtime_binding()!=(head,binding,binding_sha)
                or _sha(_read(source.PRIVATE/'train.local.npz',limit=source.MAX_ARCHIVE))!=binding['train_sha256']
                or _sha(_read(state.PRIVATE/'baseline.local.npz',limit=2*1024**2))!=binding['baseline_sha256']
                or _sha(_read(state.PRIVATE/'params.local.npz',limit=state.MAX_PARAMS))!=binding['params_sha256']):
            raise ValueError('Frozen sources changed')
        result=dict(schema='state-train-fit-result-v1',registration_commit=head,source_groups=42,
            fit_input_membership=True,student_mismatches=0,metrics=metrics,
            baseline_sha256=binding['baseline_sha256'],params_sha256=binding['params_sha256'],
            train_sha256=binding['train_sha256'],device='cpu',training=False,model_updates=0,
            new_games=0,gpu=False,independent_evaluation=False,promotion=False,
            software=dict(python='.'.join(map(str,sys.version_info[:3])),numpy=np.__version__))
        _result(previous._encoded(result),head)
        guard()
        previous._write_new(PRIVATE/'worker-result.local.json',result)
    except BaseException as exc:
        previous._write_new(PRIVATE/'worker-failed.local.json',dict(
            phase=phase,category=type(exc).__name__,retry_allowed=False))
        raise


def _validated_outputs(head,binding,guard):
    result=_result(_read(PRIVATE/'worker-result.local.json'),head)
    for key in ('baseline_sha256','params_sha256','train_sha256'):
        if result[key]!=binding[key]:
            raise ValueError('Private source identity differs')
    guard()
    return result


def run():
    if os.name!='nt' or previous.physical_available()<INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    if shutil.disk_usage(PRIVATE).free < MIN_DISK+OUTPUT_RESERVE:
        raise OSError('Private disk floor')
    head,binding,binding_sha=_runtime_binding()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    from learning.task_teacher_dev_deviation_v1.membership import token_hash
    from learning.task_pair_fresh_multi_v1.entry import owner_marker
    import psutil
    deadline=time.monotonic()+JOB['wall_seconds']+2*JOB['cleanup_seconds']
    previous._write_new(PRIVATE/'attempt.local.json',dict(
        schema='state-train-fit-attempt-v1',commit=head,retry_allowed=False))
    try:
        proc=psutil.Process(os.getpid())
        owner=previous._document(owner_marker(commit=head,binding_sha=binding_sha,pid=os.getpid(),
            created_us=int(proc.create_time()*1_000_000),deadline=deadline))
        previous._write_new(PRIVATE/'owner.local.json',owner)
        job_name='Local\\TeacherDeviation-'+secrets.token_hex(16)
        previous._write_new(PRIVATE/'claim.local.json',dict(schema='state-train-fit-claim-v1',
            commit=head,attempt=1,freeze_sha256=binding_sha,job_token_sha256=token_hash(job_name),retry_allowed=False))
        command=[sys.executable,'-B','-m','learning.task_state_train_fit_v1.experiment','child']
        job=supervisor._supervise_owned(command,str(REPO/'kaggriculture'),JOB,job_name)
        previous._write_new(PRIVATE/'job.local.json',job)
        if not transfer._job_ok(job):
            raise RuntimeError('Owned Job did not close cleanly')
        guard=lambda:previous._guard(deadline)
        _validated_outputs(head,binding,guard)
        if _registered_binding(head)[1]!=binding_sha:
            raise ValueError('Registration changed')
        names=('worker-result','job','attempt','owner','claim','child-attempt')
        receipts={name:_sha(_read(PRIVATE/(name+'.local.json'))) for name in names}
        previous._write_new(PRIVATE/'closed.local.json',dict(schema='state-train-fit-closed-v1',
            registration_commit=head,binding_sha256=binding_sha,receipts=receipts,retry_allowed=False))
        print(json.dumps(dict(status='completed')),flush=True)
    except BaseException as exc:
        previous._write_new(PRIVATE/'failed.local.json',dict(
            status='failed',category=type(exc).__name__,retry_allowed=False))
        raise


def load_completed(*,guard=lambda:None):
    """Authenticate the result; this is not a deployment or match approval."""
    if (PRIVATE/'failed.local.json').exists() or (PRIVATE/'worker-failed.local.json').exists():
        raise ValueError('Failed attempt cannot be consumed')
    closed=previous._document(_read(PRIVATE/'closed.local.json'))
    head=closed.get('registration_commit')
    binding,binding_sha=_registered_binding(head)
    names=('worker-result','job','attempt','owner','claim','child-attempt')
    if (set(closed)!={'schema','registration_commit','binding_sha256','receipts','retry_allowed'}
            or closed['schema']!='state-train-fit-closed-v1' or closed['binding_sha256']!=binding_sha
            or closed['retry_allowed'] is not False or type(closed['receipts']) is not dict
            or set(closed['receipts'])!=set(names)):
        raise ValueError('Completion seal differs')
    for name in names:
        guard()
        if _sha(_read(PRIVATE/(name+'.local.json')))!=closed['receipts'][name]:
            raise ValueError('Completion receipt differs')
    if not transfer._job_ok(previous._document(_read(PRIVATE/'job.local.json'))):
        raise ValueError('Owned Job receipt differs')
    attempt=previous._document(_read(PRIVATE/'attempt.local.json'))
    marker=previous._document(_read(PRIVATE/'child-attempt.local.json'))
    claim=previous._document(_read(PRIVATE/'claim.local.json'))
    owner=previous._document(_read(PRIVATE/'owner.local.json'))
    if (attempt!=dict(schema='state-train-fit-attempt-v1',commit=head,retry_allowed=False)
            or marker!=dict(schema='state-train-fit-child-attempt-v1',commit=head,retry_allowed=False)
            or claim.get('schema')!='state-train-fit-claim-v1' or claim.get('commit')!=head
            or type(claim.get('attempt')) is not int or claim['attempt']!=1
            or claim.get('freeze_sha256')!=binding_sha or claim.get('retry_allowed') is not False
            or type(claim.get('job_token_sha256')) is not str
            or re.fullmatch(r'[0-9a-f]{64}',claim['job_token_sha256']) is None
            or type(owner) is not dict or set(owner)!={'schema','registration_commit','binding_sha256',
                                                        'pid','created_us','deadline'}
            or owner.get('schema')!='task-pair-fresh-multi-owner-v1'
            or owner.get('registration_commit')!=head or owner.get('binding_sha256')!=binding_sha
            or type(owner.get('pid')) is not int or owner['pid']<=0
            or type(owner.get('created_us')) is not int or owner['created_us']<=0
            or type(owner.get('deadline')) not in (int,float) or not math.isfinite(owner['deadline'])):
        raise ValueError('Completion ownership differs')
    return _validated_outputs(head,binding,guard)


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self,_message):raise ValueError('Invalid command')


def main():
    parser=_PrivateArgumentParser()
    parser.add_argument('command',choices=('prepare','run','child'))
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'child':child}[args.command]()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        category=('memory' if isinstance(exc,MemoryError) else 'timeout' if isinstance(exc,TimeoutError)
                  else 'io' if isinstance(exc,OSError) else 'invalid_or_runtime')
        print(json.dumps(dict(status='failed',category=category)),file=sys.stderr)
        raise SystemExit(1) from None
