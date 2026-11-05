"""Private one-case IO adapter; authentication and Job ownership belong to caller."""
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from learning.task_anchor_once_v1.storage import owned_writes, OutputBudgetExceeded
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_pair_dev_games_v1.capture import MAX_LINE_BYTES, MAX_COMPRESSED_BYTES
from learning.task_pair_dev_games_v1.panel import GAME_LIMITS
from learning.task_request_gate_v1.policy import request_digest
from . import contract, core
from .gate import validate_target


CASE_FILE = 'case.local.json'
EVENT_FILE = 'events.jsonl.gz'
CASE_RESERVE = 1024 * 1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _error_kind(error):
    for cls, kind in ((MemoryError, 'memory'), (TimeoutError, 'timeout'),
                      (OutputBudgetExceeded, 'output_budget'), (OSError, 'io'),
                      (ValueError, 'invalid'), (KeyboardInterrupt, 'interrupted')):
        if isinstance(error, cls):
            return kind
    return 'runtime'


def _source(modules, context, source, selection):
    matches = [x for x in contract.sources() if ordered_hash(x) == ordered_hash(source)]
    contract.require(len(matches) == 1 and len(context.rows) == 32
        and type(selection) is dict and set(selection) == {'target', 'counts', 'reason', 'eligibility'}
        and selection['counts'].get('trace_lines') == 719)
    manifest = context.manifests[source['cohort']]
    panel = manifest.panel
    plans = [p for p in panel.plans() if p['game'] == source['game']]
    contract.require(len(plans) == 1)
    plan = plans[0]
    row = context.rows[(source['cohort'], source['game'])]
    contract.require(all(type(row.get(k)) is type(v) and row[k] == v for k, v in plan.items())
        and all(plan[k] == source[k] for k in ('seed_alias', 'opponent', 'side', 'policy'))
        and row.get('task_checkpoint_sha256') == panel.models[0][1]
        and row.get('crop_checkpoint_sha256') == panel.crop[1]
        and row.get('engine_sha256') == contract.ENGINE_SHA256
        and row.get('rival_sha256') == manifest.opponent(source['opponent'])[1])
    if selection['target'] is None:
        contract.require(selection['eligibility'] is None and type(selection['reason']) is str
                         and bool(selection['reason']))
    else:
        target = validate_target(selection['target'])
        contract.require(target['side'] == source['side'] and selection['reason'] is None
            and contract.WINDOW[0] <= target['step'] <= contract.WINDOW[1]
            and type(selection['eligibility']) is dict)
    root = Path(modules['fresh'].GAME)
    stage = root / 'artifacts' / modules['fresh'].STAGES[source['cohort']]
    return manifest, row, root, stage


def _assets(modules, manifest, row, root, stage, guard):
    dm = modules['driver']
    rival, rival_sha = manifest.opponent(row['opponent'])
    files = [(root / path, digest, dm.MAX_MODEL) for path, digest
             in (manifest.panel.models[0], manifest.panel.crop)]
    files += [(root / rival, rival_sha, dm.MAX_SOURCE),
              (stage / 'sources' / (row['opponent'] + '.py'), rival_sha, dm.MAX_SOURCE),
              (root / modules['worker'].WORKER_RELATIVE, manifest.panel.worker_sha, dm.MAX_SOURCE),
              (stage / 'agents' / (row['game'] + '.py'), row['bot_sha256'], dm.MAX_SOURCE)]
    for path, digest, maximum in files:
        guard()
        contract.require(dm._hash(root, path, maximum) == digest)
    guard()
    return stage / 'sources' / (row['opponent'] + '.py')


def _bounded_actor(*args, **kwargs):
    from learning.task_dagger_once_v1.policy import BoundedAgent
    return BoundedAgent(*args, **kwargs)


def _load_rival(path):
    from learning.bc_crop.audit import load
    return load(path)


def _factories(manifest, root, rival_path):
    task, task_sha = manifest.panel.models[0]
    crop, crop_sha = manifest.panel.crop

    def actor():
        return _bounded_actor(root/crop, crop_sha, root/task, task_sha,
                              limits=dict(GAME_LIMITS), trace=None)

    return actor, lambda: _load_rival(rival_path)


def trace_factory(dm, root, path, expected, guard, scans):
    """Drain each successful scan to EOF so the old same-handle hash is checked."""
    @contextmanager
    def opened():
        record = dict(lines=0, complete=False, closed=False)
        scans.append(record)
        stream = dm._trace(root, path, expected)

        def rows():
            for value in stream:
                guard()
                record['lines'] += 1
                yield value
            record['complete'] = True

        iterator = rows()
        try:
            yield iterator
            for _row in iterator:
                pass
            contract.require(record['complete'] and record['lines'] == 719)
            guard()
        finally:
            primary = sys.exc_info()[1]
            close_error = None
            for value in (iterator, stream):
                try:
                    value.close()
                except BaseException as error:
                    close_error = close_error or error
            record['closed'] = close_error is None
            if close_error is not None and primary is None:
                raise close_error

    return opened


@contextmanager
def _events(path):
    """Keep a primary model/IO error if gzip cleanup also fails."""
    raw = Path(path).open('xb')
    zipped = None
    try:
        zipped = gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0)
        yield zipped
    finally:
        primary = sys.exc_info()[1]
        close_error = None
        for stream in (zipped, raw):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as error:
                    close_error = close_error or error
        if close_error is not None and primary is None:
            raise close_error


class EventSink:
    """Persist private suffix facts; prefix data is compacted after core checks."""
    def __init__(self, stream, guard):
        self.stream, self.guard = stream, guard
        self.events = self.decoded_bytes = 0
        self.endpoints = {}
        self.turns = {arm: dict(prefix=0, suffix=0) for arm in ('control', 'intervene')}

    def __call__(self, event):
        self.guard()
        kind, arm = event['kind'], event['arm']
        contract.require(arm in self.turns and kind in ('arm_start', 'turn', 'arm_end'))
        if kind == 'turn':
            scope = event['scope']
            contract.require(scope in self.turns[arm])
            self.turns[arm][scope] += 1
            if scope == 'prefix':
                event = dict(kind=kind, arm=arm, scope=scope, step=event['step'],
                    request_sha256=request_digest(event['request']),
                    rival_request_sha256=request_digest(event['rival_request']),
                    trace_raw_ordered_sha256=ordered_hash(event['trace']),
                    stable_trace_sha256=event['stable_trace_sha256'],
                    model_counts=deepcopy(event['trace']['model_counts']),
                    turn_model_counts=deepcopy(event['trace']['turn_model_counts']))
        else:
            key = (arm, kind)
            contract.require(key not in self.endpoints)
            self.endpoints[key] = deepcopy(event['frame'])
        raw = (json.dumps(event, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        contract.require(len(raw) <= MAX_LINE_BYTES)
        self.stream.write(raw)
        self.events += 1
        self.decoded_bytes += len(raw)
        self.guard()

    def facts(self, side, start):
        return _cash_facts(self.endpoints, self.turns, side, start)


def _cash_facts(endpoints, turns, side, start):
    contract.require(set(endpoints) == {(arm, key) for arm in turns
                     for key in ('arm_start', 'arm_end')})
    result = {}
    for arm, name in (('control', 'baseline'), ('intervene', 'intervention')):
        contract.require(turns[arm] == dict(prefix=start, suffix=48))
        result[name] = {}
        for kind, endpoint, step in (('arm_start', 'start', start), ('arm_end', 'end', start+48)):
            frame = endpoints[(arm, kind)]
            contract.require(frame[0]['observation']['step'] == step)
            cash = [farm['money'] for farm in frame[0]['observation']['farms']]
            contract.require(len(cash) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in cash))
            result[name][endpoint] = dict(own=cash[side], opponent=cash[1-side])
    return result


def _write_case(dm, out, document):
    with owned_writes(out, CASE_RESERVE, allowed_paths=[CASE_FILE]):
        dm._write_new(out, out/CASE_FILE, document, limit=CASE_RESERVE)
    return dict(path=CASE_FILE, sha256=dm._hash(out, out/CASE_FILE, CASE_RESERVE),
                bytes=(out/CASE_FILE).stat().st_size)


def execute_case(modules, context, source, selection, out, guard):
    """Run one frozen point or record missing coverage; never select/retry a point."""
    contract.require(callable(guard))
    dm, out = modules['driver'], Path(out)
    guard()
    contract.require(out.is_absolute() and out.is_dir()
        and not any(os.path.lexists(out/name) for name in (CASE_FILE, EVENT_FILE)))
    manifest, old_row, root, old_stage = _source(modules, context, source, selection)
    dm._no_links(out, out)
    meter, scans, budget, sink, core_result = core.Meter(), [], None, None, None
    primary = None
    document = None
    header = dict(schema='task-harvest-window-case-v1', source=deepcopy(source),
        selection_sha256=ordered_hash(selection), target=deepcopy(selection['target']),
        source_replay_sha256=old_row['replay_gzip_sha256'], source_trace_sha256=old_row['trace_sha256'],
        task_checkpoint_sha256=manifest.panel.models[0][1], crop_checkpoint_sha256=manifest.panel.crop[1],
        retry_allowed=False, training=False, complete_games=0)
    try:
        row = {key: source[key] for key in ('role', 'opponent', 'side', 'purpose')}
        if selection['target'] is None:
            row.update(status='no_eligible_target', target=None)
            document = dict(header, passed=True, row=row, events=None)
        else:
            target = selection['target']
            rival_path = _assets(modules, manifest, old_row, root, old_stage, guard)
            replay_path = old_stage/'games'/source['game']/'replay.json.gz'
            trace_path = old_stage/'traces'/(source['game']+'.jsonl.gz')
            guard()
            replay = dm._replay(root, replay_path, old_row['replay_gzip_sha256'])
            contract.require(replay['info']['seed'] == old_row['seed']
                and [f['money'] for f in replay['steps'][0][0]['observation']['farms']] == old_row['initial_banks'])
            guard()
            actors, rivals = _factories(manifest, root, rival_path)
            budget = owned_writes(out, contract.LIMITS['per_case_output_bytes']-CASE_RESERVE,
                                  allowed_paths=[EVENT_FILE])
            with budget:
                with _events(out/EVENT_FILE) as stream:
                    sink = EventSink(stream, guard)
                    core_result = core.run_pair(replay,
                        trace_factory(dm, root, trace_path, old_row['trace_sha256'], guard, scans),
                        target=target, actor_factory=actors, rival_factory=rivals, meter=meter,
                        guard=guard, emit=sink, horizon=48)
            contract.require(core_result.get('passed') is True and len(scans) == 2
                and all(s['complete'] and s['closed'] and s['lines'] == 719 for s in scans))
            _assets(modules, manifest, old_row, root, old_stage, guard)
            contract.require(dm._hash(root, replay_path, MAX_COMPRESSED_BYTES) == old_row['replay_gzip_sha256'])
            row.update(status='paired', technical_passed=True, **sink.facts(source['side'], target['step']),
                request_changed=any(core_result['request_changed_by_offset']),
                physical_changed=any(core_result['physical_changed_by_offset']))
            events = dict(path=EVENT_FILE, sha256=dm._hash(out, out/EVENT_FILE,
                contract.LIMITS['per_case_output_bytes']), bytes=(out/EVENT_FILE).stat().st_size,
                lines=sink.events, decoded_bytes=sink.decoded_bytes, closed=True)
            document = dict(header, passed=True, row=row, events=events, core=core_result)
        guard()
    except BaseException as error:
        primary = error
        document = dict(header, passed=False, status='failed', error_kind=_error_kind(error))
        raise
    finally:
        # This reserved receipt is written even if the cooperative guard has
        # stopped work. It contains no exception message or partial paired mean.
        document = document or dict(header, passed=False, status='failed', error_kind='runtime')
        document.update(meter=meter.snapshot(), trace_scans=scans,
            output_budget=budget.snapshot() if budget is not None else None,
            emitted_events=sink.events if sink is not None else 0,
            case_record_reserved_bytes=CASE_RESERVE)
        try:
            receipt = _write_case(dm, out, document)
        except BaseException:
            if primary is None:
                raise
    return dict(document=document, receipt=receipt)


def _stable(value):
    """The old audit.stable recursion, kept pure to avoid importing its engine."""
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k != 'inference_seconds'}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def _meter_check(meter, result, start):
    contract.require(meter.get('unknown_budget_observations') == 0)
    calls = meter['calls']
    expected = dict(arm=2, actor_construct=2, rival_construct=2, restore=2,
        actor=2*(start+48), rival=2*(start+48), official_step=96,
        prefix_actor=2*start, prefix_rival=2*start,
        suffix_actor=96, suffix_rival=96, suffix_official_step=96)
    for name, amount in expected.items():
        contract.require(all(type(calls.get(name+suffix)) is int and calls[name+suffix] == amount
                             for suffix in ('_attempts', '_returns')))
    contract.require(calls.get('trace_close_failures', 0) == 0)
    for scope in ('prefix', 'suffix'):
        expected_costs = {key: sum(result[arm][scope+'_budget'][key]
            for arm in ('control', 'intervention')) for key in core.BUDGET_KEYS}
        contract.require(meter['budget_by_scope'][scope] == expected_costs)
    for arm in ('control', 'intervention'):
        value = result[arm]
        contract.require(value['prefix_turns'] == start and value['suffix_turns'] == 48
            and len(value['request_sha256']) == 48 and len(value['physical_state_sha256']) == 49
            and value['episode_budget']['controller_calls'] == start+48)
        contract.require(all(value['episode_budget'][k] == value['prefix_budget'][k]+value['suffix_budget'][k]
                             for k in core.BUDGET_KEYS))
    contract.require(result['control']['factual_states_verified'] == 49
                     and result['intervention']['factual_states_verified'] == 0)


def verify_case(modules, context, source, selection, out, guard):
    """Read-only verification; no model, official environment, or replay decode.

    A bounded extra old-trace scan authenticates stable prefix/factual-suffix
    digests. Historical full requests were checked by the frozen core; here
    prefix requests are compared between arms. Parent binds this receipt to its
    separate permit/owner/Job/acceptance chain.
    """
    contract.require(callable(guard))
    dm, parser, out = modules['driver'], modules['acceptance']._json, Path(out)
    guard()
    manifest, old_row, root, old_stage = _source(modules, context, source, selection)
    raw = dm._read(out, out/CASE_FILE, CASE_RESERVE)
    document = parser(raw, CASE_RESERVE)
    contract.require(document.get('schema') == 'task-harvest-window-case-v1'
        and document.get('passed') is True and document.get('retry_allowed') is False
        and document.get('training') is False and document.get('complete_games') == 0
        and ordered_hash(document['source']) == ordered_hash(source)
        and document['selection_sha256'] == ordered_hash(selection)
        and ordered_hash(document['target']) == ordered_hash(selection['target'])
        and document['source_replay_sha256'] == old_row['replay_gzip_sha256']
        and document['source_trace_sha256'] == old_row['trace_sha256']
        and document['task_checkpoint_sha256'] == manifest.panel.models[0][1]
        and document['crop_checkpoint_sha256'] == manifest.panel.crop[1]
        and document['case_record_reserved_bytes'] == CASE_RESERVE)
    receipt = dict(path=CASE_FILE, sha256=_sha(raw), bytes=len(raw))
    row = document['row']
    contract.require(all(type(row.get(k)) is type(source[k]) and row[k] == source[k]
                         for k in ('role', 'opponent', 'side', 'purpose')))
    if selection['target'] is None:
        contract.require(row == dict({k: source[k] for k in ('role', 'opponent', 'side', 'purpose')},
            status='no_eligible_target', target=None) and document.get('events') is None
            and document['trace_scans'] == [] and document['output_budget'] is None
            and document['emitted_events'] == 0 and document['meter'] == core.Meter().snapshot()
            and not os.path.lexists(out/EVENT_FILE))
        guard()
        return dict(document=document, receipt=receipt)
    target, result, events = selection['target'], document['core'], document['events']
    start, side = target['step'], source['side']
    contract.require(row.get('status') == 'paired' and row.get('technical_passed') is True
        and result.get('passed') is True and result.get('training') is False
        and result.get('complete_games') == result.get('extra_official_replays') == 0
        and document['meter'] == result['meter'] and len(document['trace_scans']) == 2
        and all(s == dict(lines=719, complete=True, closed=True) for s in document['trace_scans']))
    _meter_check(document['meter'], result, start)
    contract.require(events['path'] == EVENT_FILE and events.get('closed') is True
        and events['bytes'] + len(raw) <= contract.LIMITS['per_case_output_bytes']
        and (out/EVENT_FILE).stat().st_size == events['bytes']
        and dm._hash(out, out/EVENT_FILE, contract.LIMITS['per_case_output_bytes']) == events['sha256'])
    contract.require(document['output_budget']['limit_bytes'] == contract.LIMITS['per_case_output_bytes']-CASE_RESERVE
        and document['output_budget']['used_bytes'] == events['bytes']
        and document['output_budget']['refused'] == 0 and document['output_budget']['created_count'] == 1)
    old = {}
    trace_scans = []
    with trace_factory(dm, root, old_stage/'traces'/(source['game']+'.jsonl.gz'),
                       old_row['trace_sha256'], guard, trace_scans)() as traces:
        for step, trace in enumerate(traces):
            if step < start+48:
                old[step] = dict(stable_trace_sha256=ordered_hash(_stable(trace)),
                    model_counts=trace['model_counts'], turn_model_counts=trace['turn_model_counts'])
    counts = {arm: dict(prefix=0, suffix=0) for arm in ('control', 'intervene')}
    endpoints, prefix_requests, line_count, decoded_bytes = {}, {}, 0, 0
    with gzip.open(out/EVENT_FILE, 'rb') as stream:
        while True:
            guard()
            line = stream.readline(MAX_LINE_BYTES+1)
            if not line:
                break
            contract.require(len(line) <= MAX_LINE_BYTES)
            line_count += 1; decoded_bytes += len(line)
            contract.require(line_count <= events['lines'] and decoded_bytes <= events['decoded_bytes'])
            event = parser(line, MAX_LINE_BYTES)
            arm, kind = event['arm'], event['kind']
            contract.require(arm in counts and kind in ('turn', 'arm_start', 'arm_end'))
            branch = result['control' if arm == 'control' else 'intervention']
            if kind != 'turn':
                key = (arm, kind)
                contract.require(key not in endpoints)
                endpoints[key] = event['frame']
                index = 0 if kind == 'arm_start' else 48
                contract.require(core.frame_digest(event['frame'], physical=True)
                                 == branch['physical_state_sha256'][index])
                continue
            scope, step = event['scope'], event['step']
            contract.require(scope in counts[arm] and type(step) is int
                and step == counts[arm][scope] + (start if scope == 'suffix' else 0))
            counts[arm][scope] += 1
            if scope == 'prefix':
                contract.require(all(ordered_hash(event[k]) == ordered_hash(old[step][k]) for k in old[step]))
                pair = (event['request_sha256'], event['rival_request_sha256'])
                if arm == 'control':
                    prefix_requests[step] = pair
                else:
                    contract.require(prefix_requests.get(step) == pair)
            else:
                offset = step-start
                contract.require(offset < 48
                    and ordered_hash(_stable(event['trace'])) == event['stable_trace_sha256'])
                if arm == 'control':
                    contract.require(event['stable_trace_sha256'] == old[step]['stable_trace_sha256'])
                joint = [None, None]
                joint[side], joint[1-side] = event['request'], event['rival_request']
                contract.require([request_digest(x) for x in joint] == list(branch['request_sha256'][offset])
                    and core.frame_digest(event['frame'], physical=True) == branch['physical_state_sha256'][offset+1])
    contract.require(line_count == events['lines'] == document['emitted_events']
        and decoded_bytes == events['decoded_bytes']
        and dm._hash(out, out/EVENT_FILE, contract.LIMITS['per_case_output_bytes']) == events['sha256'])
    facts = _cash_facts(endpoints, counts, side, start)
    contract.require(all(row[key] == value for key, value in facts.items()))
    requests = [a != b for a, b in zip(result['control']['request_sha256'], result['intervention']['request_sha256'])]
    physical = [a != b for a, b in zip(result['control']['physical_state_sha256'], result['intervention']['physical_state_sha256'])]
    contract.require(requests == result['request_changed_by_offset'] and physical == result['physical_changed_by_offset']
        and row['request_changed'] is any(requests) and row['physical_changed'] is any(physical))
    core._same_target_proof(result['control']['target_proof'], result['intervention']['target_proof'])
    for arm, choice in (('control', target['original_choice']), ('intervention', target['harvest_choice'])):
        proof = result[arm]['target_proof']
        contract.require(all(proof[k] == target[k] for k in ('step', 'ordinal', 'input_sha256', 'original_choice', 'harvest_choice'))
            and proof['returned_choice'] == choice)
    guard()
    contract.require(dm._read(out, out/CASE_FILE, CASE_RESERVE) == raw)
    return dict(document=document, receipt=receipt)
