"""One preregistered consumed-history counterfactual screening run.

No model training or complete matches occur. Imports do not read private data;
prepare binds metadata and run requires the subsequently published commit.
"""
import argparse
import hashlib
import importlib.metadata
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
from learning.task_state_blind_diagnostic_v1 import experiment as blind
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA
from . import screen

REPO = previous.REPO
PRIVATE = REPO.parent / 'private' / 'task-teacher-counterfactual-screen-v1-20261031'
PREFIX = 'kaggriculture/learning/task_teacher_counterfactual_screen_v1/'
REGISTRATION = REPO / PREFIX / 'registration.json'
BASE_COMMIT = '09089827ffc2c42d496f766208034c45937f76ea'
CORE_SHA = '0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e'
JOB = dict(supervisor.JOB)
INITIAL_MEMORY, MIN_DISK, OUTPUT_RESERVE = 5*1024**3, 4*1024**3, 4*1024**2
CASES = tuple((role,screen.SLOTS[role],opponent,side) for role in screen.ROLES
              for opponent in screen.OPPONENTS for side in (0,1))
NEW_FILES = tuple(PREFIX + name for name in ('__init__.py','screen.py','experiment.py',
                                              'test_screen.py','test_experiment.py','REGISTRATION.zh-CN.md'))
EXTRA_RUNTIME_FILES = tuple('kaggriculture/'+name for name in (
    'bots/teacher_fertilizer_terminal_v2.py',
    'decision/action_views.py','decision/execution.py','decision/full_audit.py',
    'decision/profit_attribution.py','decision/verify_market_teacher.py',
    'learning/bc_crop/agent.py','learning/bc_crop/audit.py','learning/bc_crop/experiment.py',
    'learning/bc_crop/features.py','learning/bc_crop/prepare.py',
    'learning/bc_crop_economic_diagnostic/analysis.py',
    'learning/panel_eval_v3/experiment.py','learning/panel_eval_v3/failed_history.py',
    'learning/panel_eval_v3/launch.py','learning/panel_eval_v3/prior_v2_guard.py',
    'learning/panel_safety_v1/memory_gate.py','learning/panel_safety_v1/seed_inventory.py',
    'learning/panel_safety_v2/registered_rescan.py','learning/task_anchor_once_v1/storage.py',
    'learning/task_bc_first_water_unit_requests_v1/model_io.py',
    'learning/task_bc_first_water_unit_requests_v1/source.py',
    'learning/task_bc_holdout_v1/audit.py','learning/task_bc_holdout_v1/experiment.py',
    'learning/task_bc_holdout_v1/games.py','learning/task_bc_holdout_v1/statistics.py',
    'learning/task_bc_v1/cache.py','learning/task_bc_v1/deployment.py',
    'learning/task_bc_v1/experiment.py','learning/task_bc_v1/games.py','learning/task_bc_v1/parity.py',
    'learning/task_choice_v1/agent.py','learning/task_choice_v1/cases.py',
    'learning/task_choice_v1/dataset.py','learning/task_choice_v1/experiment.py',
    'learning/task_control_once_v1/entry.py','learning/task_dagger_once_v1/policy.py',
    'learning/task_entry_repair_v1/entry.py','learning/task_entry_repair_v1/experiment.py',
    'learning/task_job_empty_exit_v1/sampling.py','learning/task_job_sampling_race_v1/sampling.py',
    'learning/task_multi_teacher_probe_v1/teacher.py','learning/task_pair_dev_campaign_v1/acceptance.py',
    'learning/task_pair_dev_campaign_v1/campaign.py','learning/task_pair_dev_campaign_v1/manifest.py',
    'learning/task_pair_dev_games_v1/capture.py','learning/task_pair_dev_games_v1/worker.py',
    'learning/task_pair_dev_metrics_v1/consumer.py','learning/task_pair_dev_metrics_v1/panel.py',
    'learning/task_pair_dev_metrics_v1/replay_pair.py','learning/task_pair_fresh_multi_v1/prepare.py',
    'learning/task_pair_market_consumed_v1/entry.py','learning/task_reward_gate_v1/experiment.py',
    'learning/task_student_collection_v1/dataset.py','learning/task_student_collection_v1/sampling.py',
    'learning/task_student_continuation_v1/analysis.py','learning/task_student_continuation_v1/sources.py',
    'learning/task_teacher_choice_bc_v1/extract.py','learning/task_teacher_merged_eval_v1/evaluation.py',
    'learning/task_teacher_merged_eval_v1/experiment.py','learning/task_teacher_support_probe_v1/alignment.py',
    'learning/task_teacher_support_probe_v1/experiment.py','learning/task_teacher_support_probe_v1/opponent.py',
    'learning/task_teacher_support_probe_v1/terminal.py','learning/task_teacher_transfer_dataset_v1/extract.py',
    'learning/task_training_job_supervision_v1/job.py'))
SOURCE_INITIALIZERS = tuple('kaggriculture/'+name for name in (
    'decision/__init__.py','learning/bc_crop/__init__.py','learning/bc_crop_economic_diagnostic/__init__.py',
    'learning/task_anchor_once_v1/__init__.py','learning/task_bc_holdout_v1/__init__.py',
    'learning/task_bc_error_diagnostics_v1/__init__.py',
    'learning/task_bc_first_water_unit_requests_v1/__init__.py','learning/task_bc_v1/__init__.py',
    'learning/task_choice_v1/__init__.py','learning/task_control_once_v1/__init__.py',
    'learning/task_dagger_once_v1/__init__.py','learning/task_entry_repair_v1/__init__.py',
    'learning/task_multi_teacher_probe_v1/__init__.py',
    'learning/task_pair_checkpoint_v1/__init__.py','learning/task_pair_dev_games_v1/__init__.py',
    'learning/task_pair_fresh_multi_v1/__init__.py','learning/task_pair_dev_metrics_v1/__init__.py',
    'learning/task_pair_market_consumed_v1/__init__.py','learning/task_request_gate_v1/__init__.py',
    'learning/task_reward_gate_v1/__init__.py','learning/task_student_collection_v1/__init__.py',
    'learning/task_single_intervention_v2/__init__.py','learning/task_student_anchor_collection_v1/__init__.py',
    'learning/task_student_continuation_v1/__init__.py','learning/task_student_dagger_collection_v1/__init__.py',
    'learning/task_teacher_support_probe_v1/__init__.py',
    'learning/task_teacher_transfer_dataset_v1/__init__.py','learning/task_water_residual_v1/__init__.py'))
SOURCE_NAMES = tuple(dict.fromkeys(blind.SOURCE_NAMES + NEW_FILES + (
    'kaggriculture/learning/task_student_continuation_v1/compare.py',
    'kaggriculture/learning/task_student_continuation_v1/official_step.py',
    'kaggriculture/learning/task_student_dagger_collection_v1/sampling.py',
    'kaggriculture/learning/task_bc_first_water_unit_requests_v1/probe.py',
    'kaggriculture/learning/task_request_gate_v1/policy.py',
    'kaggriculture/learning/task_single_intervention_v2/runner.py',
    'kaggriculture/decision/semantics.py','kaggriculture/decision/verify_mechanisms.py')
    + EXTRA_RUNTIME_FILES + SOURCE_INITIALIZERS))


def _git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],stderr=subprocess.DEVNULL,
                                   timeout=30).decode('utf-8').strip()


def _sha(raw): return hashlib.sha256(raw).hexdigest()
def _read(path,limit=1024**2): return blind._read(path,limit=limit)
def _source_hashes(): return {name:_sha(_read(REPO/name)) for name in SOURCE_NAMES}


def _validate_software(value):
    names={'python','numpy','kaggle-environments','psutil'}
    if (type(value) is not dict or set(value)!=names
            or any(type(v) is not str or re.fullmatch(r'[A-Za-z0-9.+!_-]{1,64}',v) is None
                   for v in value.values())
            or re.fullmatch(r'\d+\.\d+\.\d+',value['python']) is None):
        raise ValueError('Runtime package version differs')
    return value


def _software():
    result=dict(python='.'.join(map(str,sys.version_info[:3])))
    for name in ('numpy','kaggle-environments','psutil'):
        result[name]=importlib.metadata.version(name)
    return _validate_software(result)


def _registration(sources,binding_sha,software):
    return dict(schema='teacher-counterfactual-screen-prereg-v1',base_commit=BASE_COMMIT,
        source_sha256=sources,private_binding_sha256=binding_sha,official_rule_commit=ENGINE_COMMIT,
        official_rule_sha256=ENGINE_SHA,official_framework_core_sha256=CORE_SHA,
        cases=[dict(role=r,batch=2,slot=t,opponent=o,side=s) for r,t,o,s in CASES],
        selection='authenticated_dagger_rank_top2_per_window_fixed_window_order_no_outcome_replacement',
        ranking_uses_historical_slots=True, windows=[list(x) for x in screen.WINDOWS],
        max_candidates_per_window=4,selected_per_window=2,max_candidate_width=256,
        max_comparisons=96,max_controller_turns_reserved=192,max_official_transitions_reserved=192,
        stop_case='first_official_canonical_python_nonaction_next_frame_difference',
        official_request_equality='existing_same_request',outside_top4='unknown',
        gate=dict(min_cases=4,required_roles=list(screen.ROLES)),
        software=_validate_software(software),job=dict(JOB),initial_memory_bytes=INITIAL_MEMORY,disk_floor_bytes=MIN_DISK,
        output_reserve_bytes=OUTPUT_RESERVE,max_attempts=1,training=False,gpu=False,
        new_games=0,independent_evaluation=False,promotion=False)


def _identity(context,guard):
    guard()
    roles=context.private_role_binding()
    selected={r:roles[r] for r in screen.ROLES}
    if (len(set(selected.values()))!=2 or context.source_role(batch=2,role='T01')!='A05'
            or context.source_role(batch=2,role='T02')!='A06'):
        raise ValueError('Completed source role mapping differs')
    _mean,_scale,metadata=context.model_normalization()
    guard()
    return dict(roles=selected,baseline_sha256=context.task_sha256,crop_sha256=context.crop_sha256,
                schema_sha256=metadata['schema_sha256'],metadata_sha256=_sha(previous._encoded(metadata)))


def _engine_identity(official_step,guard):
    """Verify installed rule and framework bytes even if no helper is called."""
    guard()
    if (getattr(official_step,'ENGINE_SHA',None)!=ENGINE_SHA
            or getattr(official_step,'ENGINE_CORE_SHA256',None)!=CORE_SHA):
        raise ValueError('Official engine identity differs')
    try:
        rule_path=Path(official_step.rules.__file__)
        core_path=Path(official_step.official_core.__file__)
    except (AttributeError,TypeError):
        raise ValueError('Official engine identity differs') from None
    if (_sha(_read(rule_path,limit=8*1024**2))!=ENGINE_SHA
            or _sha(_read(core_path,limit=8*1024**2))!=CORE_SHA):
        raise ValueError('Official engine identity differs')
    guard()


def _prepare_status():
    return all(line.startswith('?? ') and line[3:].replace('\\','/') in NEW_FILES
               for line in _git('status','--porcelain','--untracked-files=all').splitlines())


def prepare():
    if (os.name!='nt' or _git('rev-parse','HEAD')!=BASE_COMMIT or not _prepare_status()
            or PRIVATE.exists() or REGISTRATION.exists()):
        raise ValueError('Unexpected preparation state')
    if previous.physical_available()<INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    from learning.task_student_anchor_collection_v1 import sources
    deadline=time.monotonic()+300
    guard=lambda:previous._guard(deadline)
    context=sources.open_context(guard=guard)
    identity=_identity(context,guard)
    binding=dict(schema='teacher-counterfactual-screen-binding-v1',salt=secrets.token_hex(32),
                 software=_software(),**identity)
    registration=_registration(_source_hashes(),_sha(previous._encoded(binding)),binding['software'])
    PRIVATE.mkdir(parents=False,exist_ok=False)
    previous._write_new(PRIVATE/'binding.local.json',binding)
    previous._write_new(REGISTRATION,registration)
    print(json.dumps(dict(status='prepared')),flush=True)


def _registered_binding(head):
    if type(head) is not str or re.fullmatch(r'[0-9a-f]{40}',head) is None:
        raise ValueError('Invalid registration commit')
    parent=_git('rev-parse',head+'^')
    raw=subprocess.check_output(['git','-C',str(REPO),'show',
        head+':'+REGISTRATION.relative_to(REPO).as_posix()],stderr=subprocess.DEVNULL,timeout=30)
    local=_read(PRIVATE/'binding.local.json')
    binding=previous._document(local)
    keys={'schema','salt','roles','baseline_sha256','crop_sha256','schema_sha256','metadata_sha256','software'}
    if (set(binding)!=keys or binding.get('schema')!='teacher-counterfactual-screen-binding-v1'
            or type(binding.get('roles')) is not dict or set(binding['roles'])!=set(screen.ROLES)
            or any(type(x) is not int or not 0<=x<2**32 for x in binding['roles'].values())
            or len(set(binding['roles'].values()))!=2
            or any(type(binding[k]) is not str or re.fullmatch(r'[0-9a-f]{64}',binding[k]) is None
                   for k in keys-{'schema','roles','software'})):
        raise ValueError('Private binding differs')
    software=_validate_software(binding['software'])
    if parent!=BASE_COMMIT or previous._document(raw)!=_registration(_source_hashes(),_sha(local),software):
        raise ValueError('Frozen registration differs')
    return binding,_sha(local)


def _runtime_binding():
    head=_git('rev-parse','HEAD')
    if _git('status','--porcelain') or _git('rev-parse','origin/main')!=head:
        raise ValueError('Registration not clean and published')
    binding,digest=_registered_binding(head)
    return head,binding,digest


def _validate_case(coverage,plan,candidate,case):
    role,slot,opponent,side=case
    keys={'role','opponent','side','status_by_window','ranked_out_by_window',
          'source_disagreements_beyond_top4','comparisons','official_transitions_reserved','candidate_found'}
    if (type(coverage) is not dict or set(coverage)!=keys
            or (coverage['role'],coverage['opponent'],coverage['side'])!=(role,opponent,side)
            or type(coverage['side']) is not int or type(coverage['candidate_found']) is not bool
            or coverage['source_disagreements_beyond_top4']!='unknown'
            or type(plan) is not list or len(plan)!=6
            or type(coverage['status_by_window']) is not list or len(coverage['status_by_window'])!=6
            or type(coverage['ranked_out_by_window']) is not list or len(coverage['ranked_out_by_window'])!=6):
        raise ValueError('Private case shape differs')
    from learning.task_student_dagger_collection_v1.sampling import coordinate_rank
    hit=None; comparisons=transitions=0
    for window in range(6):
        selected,statuses,outside=plan[window],coverage['status_by_window'][window],coverage['ranked_out_by_window'][window]
        if (type(selected) is not list or len(selected)>2 or type(statuses) is not list or len(statuses)!=2
                or type(outside) is not int or not 0<=outside<=2 or (len(selected)<2 and outside!=0)):
            raise ValueError('Private window shape differs')
        ranks=[]
        for index in range(2):
            status=statuses[index]
            if type(status) is not str or status not in screen.STATUSES:
                raise ValueError('Private window status differs')
            if index>=len(selected):
                if status!='empty': raise ValueError('Private empty slot differs')
                continue
            row=selected[index]
            if (type(row) is not dict or set(row)!={'step','ordinal'} or type(row['step']) is not int
                    or not screen.WINDOWS[window][0]<=row['step']<=screen.WINDOWS[window][1]
                    or type(row['ordinal']) is not int or not 0<=row['ordinal']<32 or status=='empty'):
                raise ValueError('Private planned coordinate differs')
            ranks.append((coordinate_rank(slot,opponent,side,row['step'],row['ordinal']),row['step'],row['ordinal']))
            if hit is not None:
                if status!='unexamined_after_hit': raise ValueError('Private stopping rule differs')
                continue
            if status=='unexamined_after_hit': raise ValueError('Private stopping rule differs')
            comparisons+=1
            if status in ('official_next_frame_no_difference_detected','official_next_frame_different'):
                transitions+=2
            if status=='official_next_frame_different':
                hit=(window,index,row['step'],row['ordinal'])
        if ranks!=sorted(ranks) or len({r[1] for r in ranks})!=len(ranks):
            raise ValueError('Private plan rank differs')
    if (type(coverage['comparisons']) is not int or coverage['comparisons']!=comparisons
            or type(coverage['official_transitions_reserved']) is not int
            or coverage['official_transitions_reserved']!=transitions
            or coverage['candidate_found']!=(hit is not None)):
        raise ValueError('Private case accounting differs')
    if hit is None:
        if candidate is not None: raise ValueError('Unexpected private candidate')
    else:
        required={'role','slot','opponent','side','window','rank','step','ordinal',
                  'input_sha256','original_choice','teacher_choice'}
        if (type(candidate) is not dict or set(candidate)!=required
                or (candidate['role'],candidate['slot'],candidate['opponent'],candidate['side'])
                   !=(role,slot,opponent,side)
                or tuple(candidate[k] for k in ('window','rank','step','ordinal'))!=hit
                or type(candidate['input_sha256']) is not str
                or re.fullmatch(r'[0-9a-f]{64}',candidate['input_sha256']) is None
                or any(type(candidate[k]) is not str or not candidate[k] for k in ('original_choice','teacher_choice'))
                or candidate['original_choice']==candidate['teacher_choice']):
            raise ValueError('Private candidate identity differs')
    return comparisons,transitions


def _validate_meter(value,comparisons,transitions):
    keys={'comparisons_reserved','comparisons_begun','comparisons_completed','controller_turns_reserved',
          'controller_turns_actual','official_transitions_reserved','official_helpers_begun',
          'official_helpers_completed','official_transitions_confirmed_completed','incomplete_helper_step_progress'}
    if (type(value) is not dict or set(value)!=keys
            or any(type(value[k]) is not int or value[k]<0 for k in keys-{'controller_turns_actual','incomplete_helper_step_progress'})
            or value['comparisons_reserved']!=comparisons or value['comparisons_begun']!=comparisons
            or value['comparisons_completed']!=comparisons or not 0<=comparisons<=96
            or value['controller_turns_reserved']!=2*comparisons or value['controller_turns_actual']!='unknown'
            or value['official_transitions_reserved']!=transitions or transitions>192
            or value['official_helpers_begun']!=transitions//2
            or value['official_helpers_completed']!=transitions//2
            or value['official_transitions_confirmed_completed']!=transitions
            or value['incomplete_helper_step_progress']!='none'):
        raise ValueError('Private budget accounting differs')


def _documents(head,cases,meter):
    """Validate eight cases; return private plan, anonymous coverage and gate."""
    if type(cases) is not list or len(cases)!=8:
        raise ValueError('Private source case count differs')
    plans,coverage=[],[]
    comparisons=transitions=0
    for expected,case in zip(CASES,cases):
        if type(case) is not screen.PrivateCase:
            raise ValueError('Private source case differs')
        c,t=_validate_case(case.coverage,case.plan,case.candidate,expected)
        comparisons+=c;transitions+=t
        role,slot,opponent,side=expected
        plans.append(dict(role=role,slot=slot,opponent=opponent,side=side,
                          plan=case.plan,candidate=case.candidate))
        coverage.append(case.coverage)
    _validate_meter(meter,comparisons,transitions)
    found=[item for item in plans if item['candidate'] is not None]
    roles={item['role'] for item in found}
    gate=dict(candidate_cases=len(found),roles_covered=sorted(roles),
              status='sufficient_for_separate_design' if len(found)>=4 and roles==set(screen.ROLES) else 'insufficient')
    private=dict(schema='teacher-counterfactual-private-plan-v1',registration_commit=head,cases=plans)
    result=dict(schema='teacher-counterfactual-screen-result-v1',registration_commit=head,
        cases=coverage,meter=meter,gate=gate,source_games=8,trace_turns=8*719,
        training=False,gpu=False,new_games=0,independent_evaluation=False,promotion=False)
    return private,result


def _validate_documents(head,plan,result):
    if (type(plan) is not dict or set(plan)!={'schema','registration_commit','cases'}
            or plan['schema']!='teacher-counterfactual-private-plan-v1' or plan['registration_commit']!=head
            or type(plan['cases']) is not list or len(plan['cases'])!=8
            or type(result) is not dict or set(result)!={'schema','registration_commit','cases','meter','gate',
                'source_games','trace_turns','training','gpu','new_games','independent_evaluation','promotion'}
            or result['schema']!='teacher-counterfactual-screen-result-v1' or result['registration_commit']!=head
            or type(result['cases']) is not list or len(result['cases'])!=8
            or result['source_games']!=8 or result['trace_turns']!=8*719
            or result['training'] is not False or result['gpu'] is not False
            or type(result['new_games']) is not int or result['new_games']!=0
            or result['independent_evaluation'] is not False or result['promotion'] is not False):
        raise ValueError('Private result document differs')
    cases=[]
    for expected,p,c in zip(CASES,plan['cases'],result['cases']):
        role,slot,opponent,side=expected
        if (type(p) is not dict or set(p)!={'role','slot','opponent','side','plan','candidate'}
                or (p['role'],p['slot'],p['opponent'],p['side'])!=(role,slot,opponent,side)):
            raise ValueError('Private plan document differs')
        cases.append(screen.PrivateCase(c,p['candidate'],p['plan']))
    expected_plan,expected_result=_documents(head,cases,result['meter'])
    if plan!=expected_plan or result!=expected_result:
        raise ValueError('Private result accounting differs')
    return result


def _validated_outputs(head,guard):
    guard()
    plan_raw=_read(PRIVATE/'private-plan.local.json',limit=8*1024**2)
    result_raw=_read(PRIVATE/'worker-result.local.json',limit=8*1024**2)
    plan,result=previous._document(plan_raw),previous._document(result_raw)
    _validate_documents(head,plan,result)
    guard()
    return result


def child():
    head,binding,digest=_runtime_binding()
    claim=previous._document(_read(PRIVATE/'claim.local.json'))
    attempt=previous._document(_read(PRIVATE/'attempt.local.json'))
    if (claim.get('schema')!='teacher-counterfactual-screen-claim-v1' or claim.get('commit')!=head
            or type(claim.get('attempt')) is not int or claim['attempt']!=1
            or claim.get('freeze_sha256')!=digest or claim.get('retry_allowed') is not False
            or attempt!=dict(schema='teacher-counterfactual-screen-attempt-v1',commit=head,retry_allowed=False)
            or (PRIVATE/'failed.local.json').exists() or (PRIVATE/'closed.local.json').exists()):
        raise ValueError('Child is not uniquely claimed')
    from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY,verify_worker
    from learning.task_pair_fresh_multi_v1.entry import verify_live_owner
    import psutil
    owner=previous._document(_read(PRIVATE/'owner.local.json'))
    verify_worker(os.environ.get(ENV_KEY),claim,digest)
    verify_live_owner(owner,commit=head,binding_sha=digest,current_pid=os.getpid(),process=psutil.Process,
                      executable=sys.executable,samefile=os.path.samefile,now=time.monotonic())
    previous._write_new(PRIVATE/'child-attempt.local.json',dict(
        schema='teacher-counterfactual-screen-child-attempt-v1',commit=head,retry_allowed=False))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    deadline=time.monotonic()+JOB['wall_seconds']
    guard=lambda:previous._guard(deadline)
    meter=screen.Meter()
    completed,phase=0,'source'
    try:
        from learning.task_student_anchor_collection_v1 import sources
        from learning.task_student_continuation_v1 import compare,official_step
        if _software()!=binding['software']:
            raise ValueError('Runtime package version changed')
        _engine_identity(official_step,guard)
        context=sources.open_context(guard=guard)
        identity=_identity(context,guard)
        if identity!={k:v for k,v in binding.items() if k not in ('schema','salt','software')}:
            raise ValueError('Historical binding changed')
        cases=[]
        phase='screen'
        for role,slot,opponent,side in CASES:
            guard()
            with sources.read_game(context,batch=2,role=slot,opponent=opponent,side=side,guard=guard) as game:
                item=screen.run_case(game,role=role,opponent=opponent,side=side,guard=guard,
                    compare_turn=compare.compare_turn,compare_step=official_step.compare_step,meter=meter)
            cases.append(item)
            completed+=1
            del game,item
        phase='seal'
        _engine_identity(official_step,guard)
        if (_software()!=binding['software'] or _identity(context,guard)!=identity
                or _runtime_binding()!=(head,binding,digest)):
            raise ValueError('Historical sources changed')
        plan,result=_documents(head,cases,meter.snapshot())
        _validate_documents(head,plan,result)
        if shutil.disk_usage(PRIVATE).free<MIN_DISK+OUTPUT_RESERVE:
            raise OSError('Private disk floor')
        previous._write_new(PRIVATE/'private-plan.local.json',plan)
        previous._write_new(PRIVATE/'worker-result.local.json',result)
        _validated_outputs(head,guard)
    except BaseException as exc:
        previous._write_new(PRIVATE/'worker-failed.local.json',dict(
            phase=phase,category=type(exc).__name__,completed_sources=completed,
            meter=meter.snapshot(),retry_allowed=False))
        raise


def run():
    if os.name!='nt' or previous.physical_available()<INITIAL_MEMORY:
        raise MemoryError('Initial memory floor')
    if shutil.disk_usage(PRIVATE).free<MIN_DISK+OUTPUT_RESERVE:
        raise OSError('Private disk floor')
    head,binding,digest=_runtime_binding()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    from learning.task_teacher_dev_deviation_v1.membership import token_hash
    from learning.task_pair_fresh_multi_v1.entry import owner_marker
    import psutil
    deadline=time.monotonic()+JOB['wall_seconds']+2*JOB['cleanup_seconds']
    previous._write_new(PRIVATE/'attempt.local.json',dict(
        schema='teacher-counterfactual-screen-attempt-v1',commit=head,retry_allowed=False))
    try:
        proc=psutil.Process(os.getpid())
        owner=previous._document(owner_marker(commit=head,binding_sha=digest,pid=os.getpid(),
            created_us=int(proc.create_time()*1_000_000),deadline=deadline))
        previous._write_new(PRIVATE/'owner.local.json',owner)
        job_name='Local\\TeacherDeviation-'+secrets.token_hex(16)
        previous._write_new(PRIVATE/'claim.local.json',dict(schema='teacher-counterfactual-screen-claim-v1',
            commit=head,attempt=1,freeze_sha256=digest,job_token_sha256=token_hash(job_name),retry_allowed=False))
        command=[sys.executable,'-B','-m','learning.task_teacher_counterfactual_screen_v1.experiment','child']
        job=supervisor._supervise_owned(command,str(REPO/'kaggriculture'),JOB,job_name)
        previous._write_new(PRIVATE/'job.local.json',job)
        if not transfer._job_ok(job):
            raise RuntimeError('Owned Job did not close cleanly')
        guard=lambda:previous._guard(deadline)
        _validated_outputs(head,guard)
        if _registered_binding(head)[1]!=digest:
            raise ValueError('Registration changed')
        names=('private-plan','worker-result','job','attempt','owner','claim','child-attempt')
        receipts={name:_sha(_read(PRIVATE/(name+'.local.json'),limit=8*1024**2)) for name in names}
        previous._write_new(PRIVATE/'closed.local.json',dict(schema='teacher-counterfactual-screen-closed-v1',
            registration_commit=head,binding_sha256=digest,receipts=receipts,retry_allowed=False))
        print(json.dumps(dict(status='completed')),flush=True)
    except BaseException as exc:
        previous._write_new(PRIVATE/'failed.local.json',dict(status='failed',category=type(exc).__name__,
                                                             retry_allowed=False))
        raise


def load_completed(*,guard=lambda:None):
    """Recheck the private seal; never authorize matches or model changes."""
    if (PRIVATE/'failed.local.json').exists() or (PRIVATE/'worker-failed.local.json').exists():
        raise ValueError('Failed attempt cannot be consumed')
    closed=previous._document(_read(PRIVATE/'closed.local.json'))
    head=closed.get('registration_commit')
    binding,digest=_registered_binding(head)
    names=('private-plan','worker-result','job','attempt','owner','claim','child-attempt')
    if (set(closed)!={'schema','registration_commit','binding_sha256','receipts','retry_allowed'}
            or closed.get('schema')!='teacher-counterfactual-screen-closed-v1'
            or closed.get('binding_sha256')!=digest or closed.get('retry_allowed') is not False
            or type(closed.get('receipts')) is not dict or set(closed['receipts'])!=set(names)):
        raise ValueError('Completion seal differs')
    for name in names:
        guard()
        if _sha(_read(PRIVATE/(name+'.local.json'),limit=8*1024**2))!=closed['receipts'][name]:
            raise ValueError('Completion receipt differs')
    if not transfer._job_ok(previous._document(_read(PRIVATE/'job.local.json'))):
        raise ValueError('Owned Job receipt differs')
    attempt=previous._document(_read(PRIVATE/'attempt.local.json'))
    marker=previous._document(_read(PRIVATE/'child-attempt.local.json'))
    claim=previous._document(_read(PRIVATE/'claim.local.json'))
    owner=previous._document(_read(PRIVATE/'owner.local.json'))
    if (attempt!=dict(schema='teacher-counterfactual-screen-attempt-v1',commit=head,retry_allowed=False)
            or marker!=dict(schema='teacher-counterfactual-screen-child-attempt-v1',commit=head,retry_allowed=False)
            or claim.get('schema')!='teacher-counterfactual-screen-claim-v1' or claim.get('commit')!=head
            or type(claim.get('attempt')) is not int or claim['attempt']!=1
            or claim.get('freeze_sha256')!=digest or claim.get('retry_allowed') is not False
            or type(claim.get('job_token_sha256')) is not str
            or re.fullmatch(r'[0-9a-f]{64}',claim['job_token_sha256']) is None
            or type(owner) is not dict or set(owner)!={'schema','registration_commit','binding_sha256',
                                                        'pid','created_us','deadline'}
            or owner.get('schema')!='task-pair-fresh-multi-owner-v1' or owner.get('registration_commit')!=head
            or owner.get('binding_sha256')!=digest or type(owner.get('pid')) is not int or owner['pid']<=0
            or type(owner.get('created_us')) is not int or owner['created_us']<=0
            or type(owner.get('deadline')) not in (int,float) or not math.isfinite(owner['deadline'])):
        raise ValueError('Completion ownership differs')
    return _validated_outputs(head,guard)


class _PrivateArgumentParser(argparse.ArgumentParser):
    def error(self,_message): raise ValueError('Invalid command')


def main():
    parser=_PrivateArgumentParser()
    parser.add_argument('command',choices=('prepare','run','child'))
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'child':child}[args.command]()


if __name__=='__main__':
    try:
        main()
    except (Exception,KeyboardInterrupt) as exc:
        category=('memory' if isinstance(exc,MemoryError) else 'timeout' if isinstance(exc,TimeoutError)
                  else 'io' if isinstance(exc,OSError) else 'invalid_or_runtime')
        print(json.dumps(dict(status='failed',category=category)),file=sys.stderr)
        raise SystemExit(1) from None
