"""Private compact comparisons for one official counterfactual pair.

Frames are compacted independently so callers can release each replay before
opening the next. SHA-256 comparisons retain a cryptographic collision boundary
and are stricter on JSON numeric types than Python's v2 equality.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

from decision.semantics import canonical
from learning.task_bc_error_diagnostics_v1.shadow import first_difference,ordered_hash
from learning.task_request_gate_v1.policy import request_digest
from learning.task_single_intervention_v2.runner import (
    _orders, trace_without_crop_timing, crop_selection_before_target,
)
from learning.task_student_dagger_collection_v1.sources import MAX_LINE
from learning.task_student_continuation_v1.official_step import _physical


def _require(value):
    if not value: raise ValueError('Private paired comparison differs')


def _digest(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True,slots=True,repr=False)
class Compact:
    config: str
    seed: int
    frames: tuple
    actions: tuple
    initial_cash: tuple
    final_cash: tuple
    target_nonaction: object

    def __repr__(self): return '<PrivateReplayCompact>'


def compact(replay,*,target_step,guard):
    _require(callable(guard) and type(target_step) is int and 0<=target_step<719
        and type(replay) is dict and type(replay.get('steps')) is list and len(replay['steps'])==720
        and type(replay.get('configuration')) is dict and replay['configuration'].get('seed') is None
        and type(replay.get('info')) is dict and type(replay['info'].get('seed')) is int
        and 0<=replay['info']['seed']<2**32)
    frames,actions=[],[]
    for step,frame in enumerate(replay['steps']):
        guard()
        _require(type(frame) is list and len(frame)==2 and all(type(x) is dict for x in frame))
        # v2 canonical retains unit inventory order; include explicit shed order.
        frames.append((_digest(canonical(frame)),ordered_hash([_orders(frame,side) for side in (0,1)])))
        if step:
            actions.append(tuple(request_digest(frame[side]['action']) for side in (0,1)))
    final=replay['steps'][-1]
    _require(all(frame['status']=='DONE' for frame in final))
    initial=tuple(farm['money'] for farm in replay['steps'][0][0]['observation']['farms'])
    ending=tuple(farm['money'] for farm in final[0]['observation']['farms'])
    _require(len(initial)==len(ending)==2 and tuple(p['reward'] for p in final)==ending
             and all(type(x) in (int,float) and math.isfinite(x) for x in initial+ending))
    return Compact(_digest(canonical(replay['configuration'])),replay['info']['seed'],
        tuple(frames),tuple(actions),initial,ending,deepcopy(_physical(canonical(replay['steps'][target_step+1]))))


def control_matches_history(history,control):
    _require(type(history) is Compact and type(control) is Compact
        and history.seed==control.seed and history.config==control.config
        and history.frames==control.frames and history.actions==control.actions
        and history.initial_cash==control.initial_cash and history.final_cash==control.final_cash)
    return True


def intervention_matches_prefix(control,intervention,*,target_step,side):
    _require(type(control) is Compact and type(intervention) is Compact
        and type(target_step) is int and 0<=target_step<719 and type(side) is int and side in (0,1)
        and control.seed==intervention.seed and control.config==intervention.config
        and control.initial_cash==intervention.initial_cash
        and control.frames[:target_step+1]==intervention.frames[:target_step+1]
        and control.actions[:target_step]==intervention.actions[:target_step]
        and control.actions[target_step][1-side]==intervention.actions[target_step][1-side]
        and control.actions[target_step][side]!=intervention.actions[target_step][side]
        and control.target_nonaction!=intervention.target_nonaction)
    return True


def control_trace(history,current,*,target_step,ordinal,expected,guard):
    """Read both complete gzip text streams; only timing is normalized."""
    target=None
    for step in range(719):
        guard();left,right=history.readline(MAX_LINE+1),current.readline(MAX_LINE+1)
        _require(bool(left) and bool(right))
        a,b=json.loads(left),json.loads(right)
        _require(a.get('step')==b.get('step')==step
            and ordered_hash(trace_without_crop_timing(a))==ordered_hash(trace_without_crop_timing(b)))
        if step==target_step:
            _require(type(a.get('records')) is list and ordinal<len(a['records']))
            record=a['records'][ordinal]
            _require(ordered_hash(record['inputs'])==expected['input_sha256']
                and record['chosen']==expected['original_choice']
                and record['teacher_label']==expected['teacher_choice'])
            target=deepcopy(record)
    _require(not history.read(1) and not current.read(1) and target is not None)
    return target


def intervention_trace(control,treatment,*,target_step,ordinal,expected,crop_before,guard):
    _require(type(target_step) is int and 0<=target_step<719 and type(ordinal) is int and 0<=ordinal<32
             and type(crop_before) is int and crop_before>=0)
    for step in range(target_step+1):
        guard();left,right=control.readline(),treatment.readline()
        _require(bool(left) and bool(right))
        a,b=json.loads(left),json.loads(right)
        _require(a.get('step')==b.get('step')==step)
        if step<target_step:
            _require(ordered_hash(trace_without_crop_timing(a))==ordered_hash(trace_without_crop_timing(b)))
            continue
        aa,bb=trace_without_crop_timing(a),trace_without_crop_timing(b)
        ac,bc=aa['crop_records'],bb['crop_records']
        _require(crop_before<=len(ac) and crop_before<=len(bc))
        for i in range(crop_before):
            _require(ordered_hash(crop_selection_before_target(ac[i]))==
                     ordered_hash(crop_selection_before_target(bc[i])))
        ar,br=a['records'],b['records']
        _require(type(ar) is list and type(br) is list and ordinal<len(ar) and ordinal<len(br))
        for index in range(ordinal+1):
            _require(ordered_hash(ar[index]['inputs'])==ordered_hash(br[index]['inputs'])
                     and ar[index]['teacher_label']==br[index]['teacher_label'])
            if index<ordinal: _require(ar[index]['chosen']==br[index]['chosen'])
        _require(ordered_hash(ar[ordinal]['inputs'])==expected['input_sha256']
            and ar[ordinal]['chosen']==expected['original_choice']
            and br[ordinal]['chosen']==br[ordinal]['teacher_label']==expected['teacher_choice'])
    return True


def proof_pair(control,treatment,*,target_step,ordinal,expected):
    for value,mode in ((control,'control'),(treatment,'intervene')):
        _require(type(value) is dict and value.get('mode')==mode and value.get('step')==target_step
            and value.get('selection_record_index')==ordinal and value.get('teacher_label_verified') is True
            and all(value.get(key)==expected[key] for key in ('input_sha256','original_choice','teacher_choice')))
    _require(control['returned_choice']==expected['original_choice']
        and treatment['returned_choice']==expected['teacher_choice']
        and first_difference(control['budget_before'],treatment['budget_before']) is None
        and first_difference(control['budget_after'],treatment['budget_after']) is None)
    crop_before=control['budget_before']['turn']['crop_forwards']
    _require(type(crop_before) is int and crop_before>=0)
    return crop_before


def account(control,treatment,*,side):
    _require(type(control) is Compact and type(treatment) is Compact and type(side) is int and side in (0,1)
        and control.initial_cash==treatment.initial_cash
        and all(type(amount) in (int,float) and math.isfinite(amount)
                for item in (control,treatment) for amount in item.initial_cash+item.final_cash))
    def one(item):
        own=item.final_cash[side]-item.initial_cash[side]
        opp=item.final_cash[1-side]-item.initial_cash[1-side]
        margin=item.final_cash[side]-item.final_cash[1-side]
        score=1.0 if margin>0 else 0.5 if margin==0 else 0.0
        return dict(own=own,opponent=opp,margin=margin,score=score)
    a,b=one(control),one(treatment)
    delta={key:b[key]-a[key] for key in a}
    _require(all(math.isfinite(value) for value in delta.values())
        and delta['margin']==delta['own']-delta['opponent'])
    return dict(control=a,intervention=b,delta=delta)
