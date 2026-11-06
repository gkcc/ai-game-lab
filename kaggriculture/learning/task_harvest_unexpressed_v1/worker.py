"""Owned private IO for one request intervention; no model work in verification.

The original policy trace always describes original_request. The request field
is the submission, and only the intervention target may differ. Historical
prefix requests are checked against Replay by core before being compacted here.
"""
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path

from learning.task_harvest_window_v1 import worker as io
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_request_gate_v1.policy import request_digest
from learning.task_pair_dev_games_v1.capture import MAX_LINE_BYTES, MAX_COMPRESSED_BYTES
from learning.task_pair_dev_games_v1.panel import GAME_LIMITS
from learning.task_anchor_once_v1.storage import owned_writes
from . import contract, core
from .adapter import replace_harvest, _units
from .selection import validate_target, SCHEMA_VERSION as SELECTION_SCHEMA


CASE_FILE, EVENT_FILE, CASE_RESERVE = io.CASE_FILE, io.EVENT_FILE, io.CASE_RESERVE
trace_factory, _stable, _write_case = io.trace_factory, io._stable, io._write_case
_bounded_actor, _load_rival = io._bounded_actor, io._load_rival


def _source(modules, context, source, selection):
    contract.require(sum(ordered_hash(x) == ordered_hash(source) for x in contract.sources()) == 1
        and len(context.rows) == 32 and type(selection) is dict
        and set(selection) == {'schema_version', 'target', 'counts', 'reason', 'eligibility'}
        and selection['schema_version'] == SELECTION_SCHEMA
        and selection['counts'].get('trace_lines') == 719)
    manifest = context.manifests[source['cohort']]
    panel = manifest.panel
    plans = [p for p in panel.plans() if p['game'] == source['game']]
    contract.require(len(plans) == 1)
    plan, row = plans[0], context.rows[(source['cohort'], source['game'])]
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
                         and type(selection['eligibility']) is dict)
    root = Path(modules['fresh'].GAME)
    return manifest, row, root, root/'artifacts'/modules['fresh'].STAGES[source['cohort']]


def _factories(manifest, root, rival_path):
    task, task_sha = manifest.panel.models[0]
    crop, crop_sha = manifest.panel.crop

    def actor():
        return _bounded_actor(root/crop, crop_sha, root/task, task_sha,
                              limits=dict(GAME_LIMITS), trace=None)
    return actor, lambda: _load_rival(rival_path)


def _intent(trace, original, step):
    actions = _units(original)
    contract.require(type(trace) is dict and type(trace.get('step')) is int and trace['step'] == step
        and type(trace.get('records')) is list and type(trace.get('crop_records')) is list
        and all(type(trace.get(k)) is int and trace[k] == 0
                for k in ('fallbacks', 'load_errors', 'inference_errors'))
        and type(trace.get('final_reservations')) is dict
        and ordered_hash(trace['final_reservations'].get('unit_actions')) == ordered_hash(actions))
    for record in trace['records']:
        unit = record.get('unit')
        contract.require(type(unit) is int and 0 <= unit < len(actions)
            and record.get('source') == 'callback'
            and ordered_hash(record.get('unit_request')) == ordered_hash(actions[unit]))
    return actions


def _budget(total, turn, previous):
    contract.require(type(total) is dict and type(turn) is dict
        and set(total) == set(turn) == set(previous) == core.BUDGET_KEYS
        and all(type(total[k]) is int and type(turn[k]) is int
                and total[k] >= 0 and turn[k] >= 0 and total[k]-previous[k] == turn[k]
                for k in core.BUDGET_KEYS) and turn['controller_calls'] == 1)


class EventSink(io.EventSink):
    """Compact only already-verified prefix facts, retain full suffix intent."""
    def __call__(self, event):
        self.guard()
        kind, arm = event['kind'], event['arm']
        contract.require(arm in self.turns and kind in ('arm_start', 'turn', 'arm_end'))
        if kind == 'turn':
            scope = event['scope']
            contract.require(scope in self.turns[arm])
            actions = _intent(event['trace'], event['original_request'], event['step'])
            contract.require(ordered_hash(_stable(event['trace'])) == event['stable_trace_sha256'])
            self.turns[arm][scope] += 1
            if scope == 'prefix':
                contract.require(ordered_hash(event['request']) == ordered_hash(event['original_request'])
                    and event['patch'] is None and event['target_execution'] is None)
                event = dict(kind=kind, arm=arm, scope=scope, step=event['step'],
                    request_sha256=request_digest(event['request']),
                    rival_request_sha256=request_digest(event['rival_request']),
                    original_request_ordered_sha256=ordered_hash(event['original_request']),
                    submitted_request_ordered_sha256=ordered_hash(event['request']),
                    unit_requests_sha256=ordered_hash(actions),
                    trace_raw_ordered_sha256=ordered_hash(event['trace']),
                    stable_trace_sha256=event['stable_trace_sha256'], patch=None, target_execution=None,
                    model_counts=deepcopy(event['trace']['model_counts']),
                    turn_model_counts=deepcopy(event['trace']['turn_model_counts']))
        else:
            key = (arm, kind)
            contract.require(key not in self.endpoints)
            self.endpoints[key] = deepcopy(event['frame'])
        raw = (json.dumps(event, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
        contract.require(len(raw) <= MAX_LINE_BYTES)
        self.stream.write(raw)
        self.events += 1
        self.decoded_bytes += len(raw)
        self.guard()


def execute_case(modules, context, source, selection, out, guard):
    contract.require(callable(guard))
    dm, out = modules['driver'], Path(out)
    guard()
    contract.require(out.is_absolute() and out.is_dir()
        and not any(os.path.lexists(out/name) for name in (CASE_FILE, EVENT_FILE)))
    manifest, old_row, root, old_stage = _source(modules, context, source, selection)
    dm._no_links(out, out)
    meter, scans, budget, sink, primary, document = core.Meter(), [], None, None, None, None
    header = dict(schema='task-harvest-unexpressed-case-v1', source=deepcopy(source),
        selection_sha256=ordered_hash(selection), target=deepcopy(selection['target']),
        source_replay_sha256=old_row['replay_gzip_sha256'], source_trace_sha256=old_row['trace_sha256'],
        task_checkpoint_sha256=manifest.panel.models[0][1], crop_checkpoint_sha256=manifest.panel.crop[1],
        retry_allowed=False, training=False, complete_games=0)
    try:
        row = {k: source[k] for k in ('role', 'opponent', 'side', 'purpose')}
        if selection['target'] is None:
            row.update(status='no_eligible_target', target=None)
            document = dict(header, passed=True, row=row, events=None)
        else:
            target = selection['target']
            rival_path = io._assets(modules, manifest, old_row, root, old_stage, guard)
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
                with io._events(out/EVENT_FILE) as stream:
                    sink = EventSink(stream, guard)
                    result = core.run_pair(replay,
                        trace_factory(dm, root, trace_path, old_row['trace_sha256'], guard, scans),
                        target=target, first_yield_days=modules['first_yield_days'],
                        actor_factory=actors, rival_factory=rivals, meter=meter,
                        guard=guard, emit=sink, horizon=48)
            contract.require(result.get('passed') is True and len(scans) == 2
                and all(s == dict(lines=719, complete=True, closed=True) for s in scans))
            del replay
            io._assets(modules, manifest, old_row, root, old_stage, guard)
            contract.require(dm._hash(root, replay_path, MAX_COMPRESSED_BYTES) == old_row['replay_gzip_sha256'])
            row.update(status='paired', technical_passed=True, **sink.facts(source['side'], target['step']),
                request_changed=any(result['request_changed_by_offset']),
                physical_changed=any(result['physical_changed_by_offset']))
            events = dict(path=EVENT_FILE, sha256=dm._hash(out, out/EVENT_FILE,
                contract.LIMITS['per_case_output_bytes']), bytes=(out/EVENT_FILE).stat().st_size,
                lines=sink.events, decoded_bytes=sink.decoded_bytes, closed=True)
            document = dict(header, passed=True, row=row, events=events, core=result)
        guard()
    except BaseException as error:
        primary = error
        document = dict(header, passed=False, status='failed', error_kind=io._error_kind(error))
        raise
    finally:
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


def _meter_check(meter, result, start):
    io._meter_check(meter, result, start)
    for name in ('request_patch', 'suffix_request_patch'):
        contract.require(all(type(meter['calls'].get(name+suffix)) is int
            and meter['calls'][name+suffix] == 1 for suffix in ('_attempts', '_returns')))
    contract.require(all(result[arm].get('trace_rows_verified') == 719
                         for arm in ('control', 'intervention')))


def _suffix(event, branch, target, first_days, previous, old, target_facts):
    """Verify complete original intent, one submission patch and real end frame."""
    arm, step = event['arm'], event['step']
    offset, side = step-target['step'], target['side']
    original, submitted, trace = event['original_request'], event['request'], event['trace']
    _intent(trace, original, step)
    _budget(trace['model_counts'], trace['turn_model_counts'], previous)
    contract.require(ordered_hash(_stable(trace)) == event['stable_trace_sha256'])
    if arm == 'control' or offset == 0:
        contract.require(event['stable_trace_sha256'] == old[step]['stable_trace_sha256'])
    if arm == 'intervene' and offset == 0:
        changed = replace_harvest(original, target=target, trace=trace, first_yield_days=first_days)
        contract.require(ordered_hash(submitted) == ordered_hash(changed['submitted'])
                         and ordered_hash(event['patch']) == ordered_hash(changed['change']))
    else:
        contract.require(ordered_hash(submitted) == ordered_hash(original) and event['patch'] is None)
    joint = [None, None]
    joint[side], joint[1-side] = submitted, event['rival_request']
    frame = event['frame']
    contract.require([request_digest(x) for x in joint] == list(branch['request_sha256'][offset])
        and all(ordered_hash(frame[i]['action']) == ordered_hash(joint[i]) for i in (0, 1))
        and type(frame[0]['observation'].get('step')) is int
        and frame[0]['observation']['step'] == step+1
        and core.frame_digest(frame, physical=True) == branch['physical_state_sha256'][offset+1])
    if offset == 0:
        # Same full target request and stable trace, not only the selector row.
        proof = dict(mode=arm,
            **{k: deepcopy(target[k]) for k in ('side', 'step', 'ordinal', 'unit', 'target',
                'input_sha256', 'unit_requests_sha256', 'original_unit_request')},
            original_request_sha256=ordered_hash(original), stable_trace_sha256=event['stable_trace_sha256'],
            budget_before=dict(total=previous),
            budget_after=dict(total=trace['model_counts'], turn=trace['turn_model_counts']),
            patch=event['patch'])
        contract.require(ordered_hash(proof) == ordered_hash(branch['target_proof']))
        current = (ordered_hash(original), ordered_hash(_stable(trace)), ordered_hash(event['rival_request']))
        if arm == 'control':
            target_facts['original'] = current
        else:
            contract.require(target_facts['original'] == current)
        execution = core._execution([event['target_execution']], target, submitted)
        contract.require(ordered_hash(execution) == ordered_hash(branch['target_execution'])
            and ordered_hash(core._endpoint(frame, target)) == ordered_hash(branch['endpoints']['after_target']))
    else:
        contract.require(event['target_execution'] is None)


def verify_case(modules, context, source, selection, out, guard):
    """Bounded EOF consumer; no checkpoint construction, Replay decode or engine.

The core verifies full factual Replay requests/states before intervention. This
consumer verifies source trace identities, every suffix intent/submission, one
unit-only patch, ordered endpoint snapshots, accounting and execution receipts.
It does not repeat an official transition or infer success from a HARVEST name.
"""
    contract.require(callable(guard))
    dm, parser, out = modules['driver'], modules['acceptance']._json, Path(out)
    guard()
    manifest, old_row, root, old_stage = _source(modules, context, source, selection)
    raw = dm._read(out, out/CASE_FILE, CASE_RESERVE)
    document = parser(raw, CASE_RESERVE)
    contract.require(document.get('schema') == 'task-harvest-unexpressed-case-v1'
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
    receipt = dict(path=CASE_FILE, sha256=io._sha(raw), bytes=len(raw))
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
        and events['lines'] == 2*(start+48+2)
        and events['bytes']+len(raw) <= contract.LIMITS['per_case_output_bytes']
        and (out/EVENT_FILE).stat().st_size == events['bytes']
        and dm._hash(out, out/EVENT_FILE, contract.LIMITS['per_case_output_bytes']) == events['sha256'])
    contract.require(document['output_budget']['limit_bytes'] == contract.LIMITS['per_case_output_bytes']-CASE_RESERVE
        and document['output_budget']['used_bytes'] == events['bytes']
        and document['output_budget']['refused'] == 0 and document['output_budget']['created_count'] == 1)
    old, scans = {}, []
    with trace_factory(dm, root, old_stage/'traces'/(source['game']+'.jsonl.gz'),
                       old_row['trace_sha256'], guard, scans)() as traces:
        for step, trace in enumerate(traces):
            contract.require(type(trace.get('step')) is int and trace['step'] == step)
            if step < start+48:
                old[step] = dict(stable_trace_sha256=ordered_hash(_stable(trace)),
                    model_counts=trace['model_counts'], turn_model_counts=trace['turn_model_counts'],
                    unit_requests_sha256=ordered_hash(trace['final_reservations']['unit_actions']))
    counts = {arm: dict(prefix=0, suffix=0) for arm in ('control', 'intervene')}
    previous = {arm: {k: 0 for k in core.BUDGET_KEYS} for arm in counts}
    endpoints, prefix_requests, target_facts = {}, {}, {}
    line_count, decoded_bytes, arm_index, phase = 0, 0, 0, 'arm_start'
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
            contract.require(arm_index < 2 and arm == ('control', 'intervene')[arm_index])
            branch = result['control' if arm == 'control' else 'intervention']
            if phase != 'turn':
                contract.require(kind == phase)
                endpoint = 'start' if phase == 'arm_start' else 'end'
                index = 0 if phase == 'arm_start' else 48
                endpoints[(arm, kind)] = event['frame']
                contract.require(core.frame_digest(event['frame'], physical=True) == branch['physical_state_sha256'][index]
                    and ordered_hash(core._endpoint(event['frame'], target)) == ordered_hash(branch['endpoints'][endpoint]))
                if phase == 'arm_end':
                    contract.require(ordered_hash(event['result']) == ordered_hash(branch)
                                     and previous[arm] == branch['episode_budget'])
                    arm_index += 1
                    phase = 'arm_start'
                else:
                    phase = 'turn'
                continue
            scope = 'prefix' if counts[arm]['prefix'] < start else 'suffix'
            step = counts[arm][scope]+(start if scope == 'suffix' else 0)
            contract.require(kind == 'turn' and event['scope'] == scope
                             and type(event['step']) is int and event['step'] == step)
            if scope == 'prefix':
                contract.require(all(ordered_hash(event[k]) == ordered_hash(old[step][k]) for k in old[step])
                    and event['original_request_ordered_sha256'] == event['submitted_request_ordered_sha256']
                    and event['patch'] is None and event['target_execution'] is None)
                _budget(event['model_counts'], event['turn_model_counts'], previous[arm])
                pair = (event['request_sha256'], event['rival_request_sha256'], event['original_request_ordered_sha256'])
                if arm == 'control':
                    prefix_requests[step] = pair
                else:
                    contract.require(prefix_requests.get(step) == pair)
                total = event['model_counts']
            else:
                _suffix(event, branch, target, modules['first_yield_days'], previous[arm], old, target_facts)
                total = event['trace']['model_counts']
            previous[arm] = deepcopy(total)
            counts[arm][scope] += 1
            if scope == 'prefix' and counts[arm]['prefix'] == start:
                contract.require(previous[arm] == branch['prefix_budget'])
            if scope == 'suffix' and counts[arm]['suffix'] == 48:
                phase = 'arm_end'
    contract.require(arm_index == 2 and phase == 'arm_start'
        and line_count == events['lines'] == document['emitted_events']
        and decoded_bytes == events['decoded_bytes']
        and dm._hash(out, out/EVENT_FILE, contract.LIMITS['per_case_output_bytes']) == events['sha256'])
    facts = io._cash_facts(endpoints, counts, side, start)
    contract.require(all(row[k] == v for k, v in facts.items()))
    requests = [a != b for a, b in zip(result['control']['request_sha256'], result['intervention']['request_sha256'])]
    physical = [a != b for a, b in zip(result['control']['physical_state_sha256'], result['intervention']['physical_state_sha256'])]
    contract.require(requests == result['request_changed_by_offset'] and physical == result['physical_changed_by_offset']
        and row['request_changed'] is any(requests) and row['physical_changed'] is any(physical))
    core._same_target_proof(result['control']['target_proof'], result['intervention']['target_proof'])
    guard()
    contract.require(dm._read(out, out/CASE_FILE, CASE_RESERVE) == raw)
    return dict(document=document, receipt=receipt)
