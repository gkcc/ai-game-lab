"""A local, one-step diagnostic seam around the frozen production Worker.

The original run_pair cannot complete: either its original exception survives,
or the first successfully consumed suffix triggers FirstStepComplete. No old
module globals, policy, checks, storage guard or failed document are rewritten.
"""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from types import ModuleType

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_harvest_unexpressed_runtime_v2 import worker as previous
from . import capture, contract


REPO = Path(__file__).resolve().parents[3]
CASE_FILE, EVENT_FILE, CASE_RESERVE = previous.CASE_FILE, previous.EVENT_FILE, previous.CASE_RESERVE
FIELD, SCHEMA = 'first_step_diagnostic', 'task-harvest-first-step-diagnostic-v1'
REGISTRATION = 'review/evidence/task-harvest-unexpressed-runtime-v2-20261106/registration.json'
REGISTRATION_SHA256 = 'd53f43c441a376ef00b0647b8525e460e6c021a0b4499b174d7798a3fe4a614f'
SOURCE_PATHS = (
    'kaggriculture/learning/task_harvest_unexpressed_v1/core.py',
    'kaggriculture/learning/task_harvest_window_v1/core.py',
    'kaggriculture/learning/task_harvest_unexpressed_v1/worker.py',
    'kaggriculture/learning/task_harvest_window_v1/worker.py',
    'kaggriculture/learning/task_harvest_unexpressed_runtime_v2/worker.py',
    'kaggriculture/learning/task_harvest_unexpressed_v1/contract.py',
    'kaggriculture/learning/task_harvest_window_v1/gate.py',
    'kaggriculture/learning/task_request_gate_v1/policy.py',
    'kaggriculture/decision/execution.py',
)
require = contract.require


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _registry(game):
    """Authenticate a short static whitelist before entering protected writes."""
    raw = (REPO/REGISTRATION).read_bytes()
    require(_sha(raw) == REGISTRATION_SHA256)
    sources = json.loads(raw)['public_sources']
    entries = []
    for relative in SOURCE_PATHS:
        public = REPO/relative
        raw = public.read_bytes()
        require(len(raw) <= previous.MAX_SOURCE and _sha(raw) == sources[relative])
        aliases = [public]
        deployed = Path(game)/relative.removeprefix('kaggriculture/')
        if deployed.is_file() and deployed.resolve() != public.resolve():
            deployed_raw = deployed.read_bytes()
            if relative == 'kaggriculture/decision/execution.py':
                previous.preparation._execution_bytes(deployed_raw, raw, 'game')
            else:
                require(deployed_raw == raw)
            aliases.append(deployed)
        entries.append((relative, raw, aliases))
    return capture.RegisteredSources(entries)


def _capped_meter(base):
    class SingleStepMeter(base):
        CAPS = dict(arm=1, actor_construct=1, rival_construct=1, actor=241,
                    rival=241, restore=1, official_step=1, request_patch=0)

        def __init__(self):
            super().__init__()
            self.cap_refusals = 0

        def start(self, operation, scope=None):
            allowed = (operation in self.CAPS and self.calls[operation+'_attempts'] < self.CAPS[operation]
                and (operation not in ('actor', 'rival') or scope in ('prefix', 'suffix'))
                and (operation != 'official_step' or scope == 'suffix'))
            if allowed and operation in ('actor', 'rival'):
                allowed = self.calls[scope+'_'+operation+'_attempts'] < (240 if scope == 'prefix' else 1)
            if not allowed:
                self.cap_refusals += 1
                raise capture.DiagnosticCapExceeded()
            return super().start(operation, scope)

        def snapshot(self):
            value = super().snapshot()
            value['diagnostic_cap_refusals'] = self.cap_refusals
            return value
    return SingleStepMeter


def _adapt(module, collector):
    """Only this freshly compiled namespace changes; original errors propagate."""
    old_core, old_write = module.core, module._write_case
    facade = ModuleType('_first_step_case_core')
    facade.__dict__.update(vars(old_core))
    facade.Meter = _capped_meter(old_core.Meter)
    stream_counts = dict(lines=0, decoded_bytes=0)

    def run_pair(replay, trace_factory, **kwargs):
        target, sink = kwargs['target'], kwargs['emit']
        require(target['step'] == 240 and target['side'] == 0 and kwargs['horizon'] == 48)

        def emit(event):
            # It is the original sink, including its checks and write/guard.
            sink(event)
            stream_counts.update(lines=sink.events, decoded_bytes=sink.decoded_bytes)
            if event.get('kind') == 'turn' and event.get('scope') == 'suffix':
                require(event.get('arm') == 'control' and event.get('step') == 240)
                collector.suffix_sink_completed = True
                raise capture.FirstStepComplete()
        try:
            result = old_core.run_pair(replay, trace_factory, **dict(kwargs, emit=emit))
        except BaseException as error:
            try:
                collector.record(error, core=old_core, target=target, replay=replay, runtime=kwargs['runtime'])
            except BaseException:
                # Even a defective capture implementation cannot mask error.
                collector.context = None
                collector.capture = dict(status='capture_failed', bytes=0, sha256=None)
            raise
        raise capture.DiagnosticCapExceeded()  # A complete 48-step return is forbidden.

    def write_case(dm, out, document):
        require(FIELD not in document)
        value = dict(schema=SCHEMA, **collector.snapshot(), events=None,
                     original_core_completed=False, diagnostic_only=True)
        path = Path(out)/EVENT_FILE
        if path.is_file():
            value['events'] = dict(path=EVENT_FILE,
                sha256=dm._hash(out, path, contract.LIMITS['per_case_output_bytes']),
                bytes=path.stat().st_size, **stream_counts)
        document[FIELD] = value
        try:
            # Check the WHOLE receipt, including runtime preparation and legacy fields.
            capture.bounded_json(document, CASE_RESERVE-1)  # Frozen writer appends one newline.
        except capture.CaptureLimitExceeded:
            value.update(context=None, capture=dict(status='case_limit_exceeded', bytes=0, sha256=None))
            capture.bounded_json(document, CASE_RESERVE-1)
        return old_write(dm, out, document)

    facade.run_pair = run_pair
    module.core, module._write_case = facade, write_case
    return collector


def execute_case(modules, context, source, selection, out, guard):
    require(callable(guard) and source == contract.sources()[0]
        and selection['target']['step'] == 240 and selection['target']['side'] == 0)
    guard()
    module = previous._frozen_worker()
    collector = capture.Collector(_registry(modules['fresh'].GAME))
    _adapt(module, collector)
    previous._adapt(module, has_target=True, guard=guard)
    return module.execute_case(modules, context, source, selection, out, guard)


def _meter_check(meter):
    require(type(meter) is dict and meter.get('diagnostic_cap_refusals') == 0
        and type(meter['diagnostic_cap_refusals']) is int
        and meter.get('unknown_budget_observations') == 0)
    expected = dict(arm_attempts=1, actor_construct_attempts=1, actor_construct_returns=1,
        rival_construct_attempts=1, rival_construct_returns=1, actor_attempts=241,
        actor_returns=241, rival_attempts=241, rival_returns=241,
        prefix_actor_attempts=240, prefix_actor_returns=240,
        prefix_rival_attempts=240, prefix_rival_returns=240,
        suffix_actor_attempts=1, suffix_actor_returns=1,
        suffix_rival_attempts=1, suffix_rival_returns=1,
        restore_attempts=1, restore_returns=1, official_step_attempts=1,
        official_step_returns=1, suffix_official_step_attempts=1, suffix_official_step_returns=1)
    require(all(type(v) is int and v >= 0 and v == expected.get(k, 0)
                for k, v in meter['calls'].items())
        and all(meter['calls'].get(k) == v for k, v in expected.items()))
    return expected


def _preparation_check(document):
    value, expected = document.get(previous.RECEIPT_FIELD), previous._new_receipt(True)
    require(type(value) is dict and set(value) == set(expected)
        and value['schema'] == expected['schema'] and value['identity'] == expected['identity']
        and value['phase'] == expected['phase'] and value['status'] == 'prepared')
    for key, count in dict(prepare_attempts=1, prepare_returns=1, run_pair_attempts=1, run_pair_returns=0).items():
        require(type(value[key]) is int and value[key] == count)
    previous.preparation.validate_receipt(value['prepared'], production=True)


def _context_check(value, target, module, registry):
    require(type(value) is dict and value.get('schema') == SCHEMA
        and value.get('diagnostic_only') is True and value.get('original_core_completed') is False
        and type(value.get('suffix_sink_completed')) is bool
        and value.get('capture', {}).get('status') == 'complete')
    context, error = value['context'], value['error']
    require(type(context) is dict and set(context) == {'phase', 'step', 'side', 'original_request',
        'submitted_request', 'tracer_records', 'actual_after_frame', 'expected_after_frame', 'comparison',
        'target_stable_trace_sha256', 'target_model_counts', 'target_turn_model_counts',
        'interpreter_restored', 'target_request_container_kinds', 'original_type_request_comparison',
        'serialized_types_note'})
    registry.validate_location(error)
    require(error['kind'] in ('ValueError', 'TypeError', 'AssertionError', 'KeyError', 'IndexError',
                            'RuntimeError', 'FirstStepComplete'))
    require(error['location_kind'] in ('direct_raise', 'assert', 'require',
                                     'registered_callsite_only', 'controlled_stop'))
    if value['suffix_sink_completed']:
        require(error['kind'] == 'FirstStepComplete' and error['id'] == 'FIRST_SUFFIX_SINK_COMPLETED'
                and error['location_kind'] == 'controlled_stop')
    else:
        require(error['kind'] != 'FirstStepComplete' and error['location'] is not None)
    raw = capture.bounded_json(context, capture.MAX_CONTEXT_BYTES)
    require(value['capture'] == dict(status='complete', bytes=len(raw), sha256=_sha(raw))
        and context['phase'] == 'first_factual_step' and type(context['step']) is int
        and context['step'] == 240 and type(context['side']) is int and context['side'] == target['side']
        and type(context['interpreter_restored']) is bool)
    # A diagnostic may locate a genuine hook/state/request error; record results,
    # do not require those failed checks to turn green.
    actual, expected = context['actual_after_frame'], context['expected_after_frame']
    require(ordered_hash(context['comparison']) == ordered_hash(
        capture.frame_comparison(actual, expected, module.core.frame_digest)))
    require(ordered_hash(context['original_request']) == ordered_hash(context['submitted_request'])
        and type(context['tracer_records']) is list
        and len(context['tracer_records']) <= capture.MAX_TRACE_RECORDS)
    kinds = context['target_request_container_kinds']
    require(type(kinds) is dict and set(kinds) == {'original', 'submitted', 'actual_accepted'}
        and all(k in capture.CONTAINER_KINDS for k in kinds.values()))
    check = context['original_type_request_comparison']
    require(type(check) is dict and set(check) == {'outcome', 'error_kind'}
        and check['outcome'] in ('equal', 'different', 'raises')
        and (check['error_kind'] is None if check['outcome'] != 'raises'
             else check['error_kind'] in ('ValueError', 'TypeError', 'AssertionError', 'RuntimeError', 'OtherError')))
    require(context['serialized_types_note'] ==
        'JSON retains content/order, not Python dict subclasses; not proof same_request passed')
    return context, error


def _verify_events(module, dm, out, events, old, target, context, completed, guard):
    require(type(events) is dict and events['path'] == EVENT_FILE
        and type(events['lines']) is int and events['lines'] == 241+int(completed)
        and type(events['decoded_bytes']) is int and 0 < events['decoded_bytes'] <= 16*1024**2
        and type(events['bytes']) is int and 0 < events['bytes'] <= contract.LIMITS['per_case_output_bytes']
        and (out/EVENT_FILE).stat().st_size == events['bytes']
        and dm._hash(out, out/EVENT_FILE, contract.LIMITS['per_case_output_bytes']) == events['sha256'])
    previous_counts = {k: 0 for k in module.core.BUDGET_KEYS}
    lines, decoded = 0, 0
    with gzip.open(out/EVENT_FILE, 'rb') as stream:
        while True:
            guard()
            raw = stream.readline(module.MAX_LINE_BYTES+1)
            if not raw:
                break
            lines += 1
            decoded += len(raw)
            require(lines <= events['lines'] and len(raw) <= module.MAX_LINE_BYTES
                and decoded <= events['decoded_bytes'])
            event = json.loads(raw)
            require(event['arm'] == 'control')
            if lines == 1:
                require(event['kind'] == 'arm_start' and event['step'] == 240)
                # Existing frozen frame representation must be parseable.
                module.core.frame_digest(event['frame'])
                continue
            step = lines-2
            require(event['kind'] == 'turn' and type(event['step']) is int and event['step'] == step)
            if step < 240:
                require(event['scope'] == 'prefix' and event['patch'] is None and event['target_execution'] is None
                    and event['original_request_ordered_sha256'] == event['submitted_request_ordered_sha256']
                    and all(ordered_hash(event[k]) == ordered_hash(old[step][k]) for k in old[step]))
                module._budget(event['model_counts'], event['turn_model_counts'], previous_counts)
                previous_counts = event['model_counts']
            else:
                require(completed and step == 240 and event['scope'] == 'suffix' and event['patch'] is None
                    and ordered_hash(event['request']) == ordered_hash(context['submitted_request'])
                    and ordered_hash(event['original_request']) == ordered_hash(context['original_request'])
                    and ordered_hash(event['frame']) == ordered_hash(context['actual_after_frame'])
                    and event['stable_trace_sha256'] == context['target_stable_trace_sha256'])
                module._intent(event['trace'], event['original_request'], 240)
                require(ordered_hash(module._stable(event['trace'])) == event['stable_trace_sha256'])
    require(lines == events['lines'] and decoded == events['decoded_bytes']
        and dm._hash(out, out/EVENT_FILE, contract.LIMITS['per_case_output_bytes']) == events['sha256'])
    module._budget(context['target_model_counts'], context['target_turn_model_counts'], previous_counts)
    require(context['target_model_counts'] == old[240]['model_counts']
        and context['target_turn_model_counts'] == old[240]['turn_model_counts']
        and context['target_stable_trace_sha256'] == old[240]['stable_trace_sha256']
        and ordered_hash(module._units(context['original_request'])) == old[240]['unit_requests_sha256'])
    return previous_counts


def verify_case(modules, context, source, selection, out, guard):
    """Read-only narrow consumer: no extra model, Replay decode or transition."""
    require(callable(guard) and source == contract.sources()[0]
        and selection['target']['step'] == 240 and selection['target']['side'] == 0)
    guard()
    module, dm, out = previous._frozen_worker(), modules['driver'], Path(out)
    manifest, old_row, root, stage = module._source(modules, context, source, selection)
    dm._no_links(out, out)
    raw = dm._read(out, out/CASE_FILE, CASE_RESERVE)
    document = json.loads(raw)
    require(document.get('schema') == 'task-harvest-unexpressed-case-v1'
        and document.get('passed') is False and document.get('status') == 'failed'
        and document.get('error_kind') in ('invalid', 'runtime')
        and ordered_hash(document['source']) == ordered_hash(source)
        and ordered_hash(document['target']) == ordered_hash(selection['target'])
        and document['selection_sha256'] == ordered_hash(selection)
        and document['source_replay_sha256'] == old_row['replay_gzip_sha256']
        and document['source_trace_sha256'] == old_row['trace_sha256']
        and document['task_checkpoint_sha256'] == manifest.panel.models[0][1]
        and document['crop_checkpoint_sha256'] == manifest.panel.crop[1]
        and document['case_record_reserved_bytes'] == CASE_RESERVE
        and document.get('retry_allowed') is False and document.get('training') is False
        and type(document.get('complete_games')) is int and document['complete_games'] == 0)
    _preparation_check(document)
    counts = _meter_check(document['meter'])
    value = document[FIELD]
    private, error = _context_check(value, selection['target'], module, _registry(root))
    events, completed = value['events'], value['suffix_sink_completed']
    require(document['emitted_events'] == events['lines']
        and document['trace_scans'] == [dict(lines=241, complete=False, closed=True)])
    budget = document['output_budget']
    require(budget['limit_bytes'] == contract.LIMITS['per_case_output_bytes']-CASE_RESERVE
        and budget['used_bytes'] == events['bytes'] and budget['refused'] == 0
        and budget['created_count'] == 1 and events['bytes']+len(raw) <= contract.LIMITS['per_case_output_bytes'])
    old, scans = {}, []
    with module.trace_factory(dm, root, stage/'traces'/(source['game']+'.jsonl.gz'),
                              old_row['trace_sha256'], guard, scans)() as rows:
        for step, row in enumerate(rows):
            require(type(row.get('step')) is int and row['step'] == step)
            if step <= 240:
                old[step] = dict(stable_trace_sha256=ordered_hash(module._stable(row)),
                    model_counts=row['model_counts'], turn_model_counts=row['turn_model_counts'],
                    unit_requests_sha256=ordered_hash(row['final_reservations']['unit_actions']))
    require(scans == [dict(lines=719, complete=True, closed=True)])
    prefix = _verify_events(module, dm, out, events, old, selection['target'], private, completed, guard)
    require(document['meter']['budget_by_scope'] == dict(prefix=prefix, suffix=private['target_turn_model_counts']))
    guard()
    require(dm._read(out, out/CASE_FILE, CASE_RESERVE) == raw)
    receipt = dict(path=CASE_FILE, sha256=_sha(raw), bytes=len(raw))
    diagnostic = dict(passed=True, status='diagnostic_collected',
        outcome='first_step_complete' if completed else 'located_original_error',
        error_id=error['id'], location_kind=error['location_kind'], location=deepcopy(error['location']),
        rejected_value_kind=error['rejected_value_kind'], counts=counts,
        full_horizon_verified=False, accepted_pairs=0, complete_games=0, training=False,
        comparison=dict(semantic_frame_equal=private['comparison']['actual_semantic_sha256'] ==
            private['comparison']['expected_semantic_sha256'],
            ordered_frame_equal=private['comparison']['actual_ordered_sha256'] ==
            private['comparison']['expected_ordered_sha256'],
            original_type_request_check=deepcopy(private['original_type_request_comparison']),
            accepted_action_container_kind=private['target_request_container_kinds']['actual_accepted']),
        verification_trace_rows=719, execution_trace_rows=241, event_lines=events['lines'])
    return dict(document=document, receipt=receipt, diagnostic=diagnostic)
