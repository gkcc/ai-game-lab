"""One post-hoc selector call, with complete streamed trace integrity checks."""
import json
from pathlib import Path
import time
import numpy as np
from learning.task_action_impact_diagnostic_v1.analysis import Meter, decoded_chunks, compare_callbacks
from learning.task_action_observation_v1.analysis import raw_path
from learning.task_bc_v1.model import TaskNetwork
from .experiment import OUT, RAW_PARENT, PARENT, ROOT, ready, verify_commit, deadline_check, read, sha, write_new

LIMITS = dict(raw_files=2,decode_passes_per_file=1,compressed_bytes=128*1024**2,
    decoded_bytes=1024**3,max_line_bytes=16*1024**2,max_candidates=8192,max_cpu_forwards=1,seconds=300)


class ReceiptFailure(RuntimeError):
    def __init__(self, primary_error_kind, receipt_error_kind):
        super().__init__('Probe receipt failed')
        self.primary_error_kind,self.receipt_error_kind = primary_error_kind,receipt_error_kind


def selected_line(path, identity, meter, check, max_line_bytes):
    target, line, pending = None, 0, bytearray()
    with decoded_chunks(path,identity,meter) as chunks:
        for chunk in chunks:
            check()
            pending.extend(chunk)
            while True:
                end = pending.find(b'\n')
                if end < 0: break
                if end == 0 or end > max_line_bytes: raise ValueError('Invalid trace line length')
                if line == 2: target = json.loads(bytes(pending[:end]))
                del pending[:end+1]
                line += 1
            if len(pending) > max_line_bytes: raise ValueError('Trace line budget exceeded')
    if pending or line != 719 or target is None: raise ValueError('Invalid trace framing')
    return target


def matched_records(candidate, teacher):
    comparison = compare_callbacks(candidate,teacher)
    choice = comparison['choice']
    if choice is None: raise ValueError('No matching choice difference')
    i = choice['callback']
    a,b = candidate[i],teacher[i]
    if (a['source'] != 'callback' or b['source'] != 'teacher'
        or a['op'] != 'PLANT' or b['op'] != 'PLANT' or a['unit'] == b['unit']
        or a['teacher_label'] != b['chosen']):
        raise ValueError('Registered choice characteristics changed')
    return a,b


def ranking(inputs, candidate, teacher, logits):
    candidates = inputs['candidates']
    if not isinstance(candidates,list) or not 1 <= len(candidates) <= LIMITS['max_candidates']:
        raise ValueError('Candidate budget exceeded')
    ids = [c['id'] for c in candidates]
    if any(not isinstance(x,str) for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('Invalid candidate identities')
    scores = np.asarray(logits,dtype=np.float64)
    if scores.shape != (len(ids),) or not np.isfinite(scores).all(): raise ValueError('Invalid logits')
    if candidate['chosen'] not in ids or teacher['chosen'] not in ids: raise ValueError('Missing selected candidate')
    chosen, reference = ids.index(candidate['chosen']), ids.index(teacher['chosen'])
    if int(np.argmax(scores)) != chosen: raise ValueError('Selected argmax changed')
    return dict(candidate_count=len(ids),teacher_rank=1+int(np.count_nonzero(scores > scores[reference]))
        +int(np.count_nonzero(scores[:reference] == scores[reference])),
        selected_minus_teacher_score=float(scores[chosen]-scores[reference]),
        selected_teacher_exact_tie=bool(scores[chosen] == scores[reference]),
        teacher_score_tie_count=int(np.count_nonzero(scores == scores[reference])),
        same_target=bool(candidates[chosen]['target'] == candidates[reference]['target']))


def run(out=OUT):
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    if (cfg['seed'],cfg['opponent'],cfg['side'],cfg['step']) != (3342200684,'round2',0,2) or cfg['limits'] != LIMITS:
        raise ValueError('Probe scope changed')
    if (out/'probe-started.json').exists(): raise ValueError('Attempt already consumed')
    marker = dict(created=False,complete=False)
    meter, started = None,time.monotonic()
    forward_started,forward_completed = False,False
    def check():
        deadline_check()
        if time.monotonic()-started > LIMITS['seconds']: raise TimeoutError('Probe budget exceeded')
    try:
        marker_path = (out/'probe-started.json').resolve()
        if not marker_path.is_relative_to(OUT): raise ValueError('Marker outside experiment')
        with marker_path.open('x',encoding='utf-8',newline='\n') as stream:
            marker['created'] = True
            json.dump(dict(attempt=1,retry_allowed=False,freeze_sha256=sha(out/'freeze.json')),stream,indent=2)
            stream.write('\n')
        marker['complete'] = True
        meter = Meter(LIMITS['compressed_bytes'],LIMITS['decoded_bytes'])
        traces = {}
        for arm in ('candidate','teacher_task'):
            check()
            ident = cfg['raw_inputs'][arm]
            traces[arm] = selected_line(raw_path(RAW_PARENT,ident,'trace'),ident,meter,check,LIMITS['max_line_bytes'])
            if traces[arm]['step'] != 2: raise ValueError('Unexpected trace step')
            expected = cfg['candidate_checkpoint_sha256'] if arm == 'candidate' else None
            if traces[arm]['task_checkpoint_sha256'] != expected or traces[arm]['checkpoint_sha256'] != cfg['crop_checkpoint_sha256']:
                raise ValueError('Trace model identity changed')
        snapshot = meter.snapshot()
        if snapshot['files_opened'] != 2 or snapshot['files_completed'] != 2 or snapshot['raw_decode_passes_per_file'] != 1:
            raise ValueError('Incomplete trace integrity')
        a,b = matched_records(traces['candidate']['records'],traces['teacher_task']['records'])
        inputs = a['inputs']
        candidates = inputs['candidates']
        if not isinstance(candidates,list) or not 1 <= len(candidates) <= LIMITS['max_candidates']:
            raise ValueError('Candidate budget exceeded')
        check()
        path = (ROOT/cfg['candidate_checkpoint']).resolve()
        if not path.is_relative_to(ROOT.resolve()): raise ValueError('Model path outside root')
        model = TaskNetwork(path,cfg['candidate_checkpoint_sha256'])
        if sha(path) != cfg['candidate_checkpoint_sha256']:
            raise ValueError('Model identity changed after load')
        check()
        forward_started = True
        chosen = model(inputs)
        forward_completed = True
        if sha(path) != cfg['candidate_checkpoint_sha256']:
            raise ValueError('Model identity changed after forward')
        if chosen != a['chosen'] or model.last['chosen'] != chosen: raise ValueError('Candidate selection changed')
        result = ranking(inputs,a,b,model.last['logits'])
        check()
        output = dict(passed=True,**result,raw_reads=snapshot,complete_trace_stream_decode=True,
            parsed_trace_lines_per_file=1,cpu_forwards=1,official_transitions=0,new_games=0,training_updates=0,
            freeze_sha256=sha(out/'freeze.json'),parent_freeze_sha256=sha(PARENT/'freeze.json'),method_sha256=sha(Path(__file__)),
            parent_analysis_sha256=sha(PARENT/'analysis-result.json'),
            trace_sha256={arm:cfg['raw_inputs'][arm]['sha256'] for arm in ('candidate','teacher_task')},
            model_identity_check_limit='Repeated hashes detect persistent change, not a transient replacement restored between checks',
            interpretation='Post-hoc selected case; no generalization, feature causality or financial causality established',retry_allowed=False)
        write_new(out/'probe-result.json',output)
        return output
    except BaseException as exc:
        if marker['created']:
            try:
                write_new(out/'probe-failure.local.json',dict(error_kind=type(exc).__name__,marker=marker,
                    cpu_forward_started=forward_started,cpu_forward_completed=forward_completed,
                    cpu_forward_count=(1 if forward_completed else 'unknown' if forward_started else 0),
                    raw_reads=meter.snapshot() if meter else None,retry_allowed=False,partial_work_not_refunded=True))
            except BaseException as receipt:
                raise ReceiptFailure(type(exc).__name__,type(receipt).__name__) from None
        raise
