"""Single-pass saved-data diagnosis. No simulation, network or controller calls.

The two-player replay is an offline audit source, never a deployment input.
Only the existing pure visibility adapter is used to check recorded actor inputs.
Changed requests, observed effects and completion remain distinct quantities.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import time
import zlib

from decision.api import observe
from decision.semantics import digest
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash

ARMS = ('baseline', 'candidate', 'teacher_task')
OPS = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
MOVES = ('NORTH', 'SOUTH', 'EAST', 'WEST')
PARTITIONS = ('no_task_disagreement', 'task_disagreement_same_request',
              'request_difference_no_recorded_observation_effect',
              'observation_difference_zero_terminal_margin_gap',
              'observation_difference_nonzero_terminal_margin_gap', 'unknown')


def typed_request(value):
    """Frozen request semantics, with stdlib finite check instead of NumPy."""
    kind = type(value)
    if kind is dict:
        if any(type(k) is not str for k in value):
            raise ValueError('Non-string request key')
        return ['dict', [[k, typed_request(value[k])] for k in sorted(value)]]
    if kind is list:
        return ['list', [typed_request(v) for v in value]]
    if kind is float:
        if not math.isfinite(value):
            raise ValueError('Nonfinite request')
        return ['float', 0. if value == 0 else value]
    if kind in (str, int, bool, type(None)):
        return [kind.__name__, value]
    raise ValueError('Non-JSON request')


def request_digest(value):
    raw = json.dumps(typed_request(value), separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def physical_digest(frame):
    # The wrapper is material to the inventory-order-aware semantic helper.
    return digest([{'observation': p['observation']} for p in frame])


class Meter:
    def __init__(self, compressed_limit=2**30, decoded_limit=32*2**30):
        self.compressed_limit, self.decoded_limit = compressed_limit, decoded_limit
        self.compressed = self.decoded = 0
        self.opened, self.receipts = set(), []

    def snapshot(self):
        return dict(compressed_input_read_bytes=self.compressed,
                    decoded_input_bytes=self.decoded, files_opened=len(self.opened),
                    files_completed=len(self.receipts), raw_decode_passes_per_file=1)


class HashReader:
    """No seek, no separate hash pass, and a bound before every actual read."""
    def __init__(self, path, identity, meter):
        key = str(path.resolve())
        if key in meter.opened:
            raise ValueError('Second raw read refused')
        if path.stat().st_size != identity['size']:
            raise ValueError('Raw size changed')
        meter.opened.add(key)
        self.stream, self.identity, self.meter = path.open('rb', buffering=0), identity, meter
        self.count, self.hash = 0, hashlib.sha256()

    def read(self, n=-1):
        remaining_file = self.identity['size'] - self.count
        if remaining_file == 0:
            return b''
        remaining_budget = self.meter.compressed_limit - self.meter.compressed
        if remaining_budget <= 0:
            raise ValueError('Compressed budget exhausted')
        n = min(remaining_file, remaining_budget, 65536 if n < 0 else n)
        data = self.stream.read(n)
        if not data:
            raise ValueError('Unexpected compressed EOF')
        self.count += len(data)
        self.meter.compressed += len(data)
        self.hash.update(data)
        return data

    def finish(self, decoded):
        if self.count != self.identity['size'] or self.hash.hexdigest() != self.identity['sha256']:
            raise ValueError('Raw identity mismatch')
        self.meter.receipts.append(dict(path=self.identity['path'], compressed_bytes=self.count,
            decoded_bytes=decoded, sha256=self.hash.hexdigest(), decode_passes=1))


@contextmanager
def decoded_chunks(path, identity, meter):
    source = HashReader(path, identity, meter)
    decoded = 0
    exhausted = False
    try:
        def chunks():
            nonlocal decoded, exhausted
            decoder, pending = None, b''
            while True:
                if not pending:
                    pending = source.read(65536)
                    if not pending:
                        if decoder is not None:
                            raise ValueError('Truncated gzip member')
                        exhausted = True
                        return
                if decoder is None:
                    decoder = zlib.decompressobj(wbits=31)
                available = meter.decoded_limit - meter.decoded
                if available <= 0:
                    raise ValueError('Decoded budget exhausted')
                # max_length bounds actual inflater output, not just returned
                # BufferedReader bytes. Each concatenated gzip member is checked.
                block = decoder.decompress(pending, max_length=min(65536, available))
                pending = decoder.unused_data if decoder.eof else decoder.unconsumed_tail
                if decoder.eof:
                    decoder = None
                meter.decoded += len(block)
                decoded += len(block)
                if block:
                    yield block
        yield chunks()
        if not exhausted:
            raise ValueError('Raw stream was not consumed to EOF')
        source.finish(decoded)
    finally:
        source.stream.close()


def read_json(path, identity, meter):
    with decoded_chunks(path, identity, meter) as chunks:
        return json.loads(b''.join(chunks))


def trace_lines(path, identity, meter):
    with decoded_chunks(path, identity, meter) as chunks:
        pending = bytearray()
        for block in chunks:
            pending.extend(block)
            start = 0
            while True:
                end = pending.find(b'\n', start)
                if end < 0:
                    break
                if end == start:
                    raise ValueError('Empty trace record')
                yield json.loads(pending[start:end])
                start = end + 1
            if start:
                del pending[:start]
        if pending:
            raise ValueError('Trace lacks its registered newline terminator')


def compact_replay(value, game):
    frames, side = value['steps'], game['side']
    if len(frames) != 720 or any(len(f) != 2 for f in frames):
        raise ValueError('Expected complete two-player replay')
    result = dict(full=[], physical=[], actor_observation=[], configuration=[],
                  requests=[], components=[], units=[], banks=[], days=[], positions=[])
    for step, frame in enumerate(frames):
        if frame[0]['observation']['step'] != step:
            raise ValueError('Replay step mismatch')
        result['full'].append(digest(frame))
        result['physical'].append(physical_digest(frame))
        farms = frame[0]['observation']['farms']
        result['banks'].append([f['money'] for f in farms])
        if step == 719:
            continue
        obs = deepcopy(frame[side]['observation'])
        for name in ('step', 'farms', 'market', 'town', 'day', 'hour'):
            obs[name] = deepcopy(frame[0]['observation'][name])
        view = observe(obs, value['configuration'])
        result['actor_observation'].append(ordered_hash(view.observation))
        result['configuration'].append(ordered_hash(view.configuration))
        result['days'].append(view.observation['day'])
        own = view.observation['farms'][side]
        result['positions'].append([own['farmer'], *own['hands']])
        actions = [s['action'] for s in frames[step + 1]]
        if set(actions[side]) != {'farmer', 'hands', 'market'}:
            raise ValueError('Incomplete actor request')
        result['requests'].append([request_digest(a) for a in actions])
        result['components'].append({k: request_digest(actions[side][k])
                                     for k in ('farmer', 'hands', 'market')})
        result['units'].append([request_digest(a) for a in
                                [actions[side]['farmer'], *actions[side]['hands']]])
    expected = game['expected']
    terminal = result['banks'][-1]
    checks = dict(banks=terminal, bank=terminal[side], opponent_bank=terminal[1-side],
        margin=terminal[side]-terminal[1-side],
        net_profit=terminal[side]-result['banks'][0][side],
        opponent_net_profit=terminal[1-side]-result['banks'][0][1-side],
        rewards=[s['reward'] for s in frames[-1]], status=[s['status'] for s in frames[-1]])
    checks['score'] = 1. if terminal[side] > terminal[1-side] else .5 if terminal[side] == terminal[1-side] else 0.
    for k, v in checks.items():
        if v != expected[k]:
            raise ValueError('Terminal metadata mismatch: '+k)
    if checks['status'] != ['DONE', 'DONE']:
        raise ValueError('Replay is unfinished')
    result['terminal'] = checks
    return result


def request_kind(request):
    op = request[0] if type(request) is list and request else None
    return 'move' if op in MOVES else 'work' if op in OPS else 'other'


def validate_trace(trace, replay, step, descriptor):
    if trace['step'] != step or any(trace[k] != 0 for k in ('load_errors', 'inference_errors', 'fallbacks')):
        raise ValueError('Trace identity or error count mismatch')
    records = trace['records']
    if type(records) is not list:
        raise ValueError('Missing task records')
    current = {}
    for record in records:
        inputs = record['inputs']
        for k in ('shared_x', 'candidates', 'planned', 'observation', 'configuration',
                  'inventory_item_order', 'shed_item_order'):
            inputs[k]
        for k in ('seeds', 'needed_seeds', 'supply', 'projected_shed', 'unit_actions', 'available_units'):
            inputs['planned'][k]
        if ordered_hash(inputs['observation']) != replay['actor_observation'][step]:
            raise ValueError('Trace does not match actor-visible replay projection')
        if ordered_hash(inputs['configuration']) != replay['configuration'][step]:
            raise ValueError('Trace configuration mismatch')
        if inputs['inventory_item_order'] != [list(i) for i in inputs['observation']['private']['inventories']]:
            raise ValueError('Recorded inventory order mismatch')
        if inputs['shed_item_order'] != list(inputs['observation']['private']['shed']):
            raise ValueError('Recorded shed order mismatch')
        candidates = inputs['candidates']
        for candidate in candidates:
            for k in ('id', 'unit', 'target', 'op', 'x'):
                candidate[k]
        ids = [c['id'] for c in candidates]
        if not ids or len(set(ids)) != len(ids) or record['teacher_label'] not in ids:
            raise ValueError('Invalid candidate identities')
        chosen = [c for c in candidates if c['id'] == record['chosen']]
        if len(chosen) != 1:
            raise ValueError('Missing selected candidate')
        chosen = chosen[0]
        for k in ('unit', 'target', 'op'):
            if record[k] != chosen[k]:
                raise ValueError('Selected descriptor mismatch')
        unit = record['unit']
        if type(unit) is not int or not 0 <= unit < len(replay['units'][step]):
            raise ValueError('Unknown selected unit')
        if request_digest(record['unit_request']) != replay['units'][step][unit]:
            raise ValueError('Trace emitted unit request differs from replay')
        if unit in current or record['op'] not in OPS:
            raise ValueError('Duplicate or unsupported task assignment')
        kind = request_kind(record['unit_request'])
        descriptor['selected_ops'][record['op']] += 1
        descriptor['emitted_request_kind'][kind] += 1
        current[unit] = dict(target=record['target'], op=record['op'], kind=kind,
                             position=replay['positions'][step][unit])
    # Only adjacent-turn, same-day geometry. No assertion of noncompletion.
    if descriptor['previous_day'] == replay['days'][step]:
        for unit in current.keys() & descriptor['previous'].keys():
            a, b = descriptor['previous'][unit], current[unit]
            if (a['target'], a['op']) != (b['target'], b['op']):
                descriptor['adjacent_selected_identity_changes'] += 1
                if a['kind'] == 'move':
                    descriptor['identity_changes_after_move_request'] += 1
    descriptor['previous'], descriptor['previous_day'] = current, replay['days'][step]
    descriptor['trace_turns'] += 1
    return records


def compare_callbacks(left, right):
    """Align one turn only until the FIRST unequal input OR choice."""
    result = dict(matched_context_calls=0, choice=None, context_stop=None)
    for index, (a, b) in enumerate(zip(left, right)):
        if ordered_hash(a['inputs']) != ordered_hash(b['inputs']):
            result['context_stop'] = 'input_difference'
            break
        result['matched_context_calls'] += 1
        if a['chosen'] != b['chosen']:
            result['choice'] = dict(callback=index, candidate_op=a['op'], reference_op=b['op'],
                same_unit=a['unit'] == b['unit'], same_target=a['target'] == b['target'],
                candidate_request_kind=request_kind(a['unit_request']),
                reference_request_kind=request_kind(b['unit_request']),
                emitted_unit_request_equal=request_digest(a['unit_request']) == request_digest(b['unit_request']))
            break
    else:
        if len(left) != len(right):
            result['context_stop'] = 'callback_count_difference'
    return result


def first_request_difference(a, b, side):
    for step in range(719):
        if a['full'][step] != b['full'][step]:
            raise ValueError('Full history diverged before first request difference')
        if a['requests'][step][1-side] != b['requests'][step][1-side]:
            raise ValueError('Opponent request differed on the common history prefix')
        if a['requests'][step][side] != b['requests'][step][side]:
            return step
    if a['full'][-1] != b['full'][-1]:
        raise ValueError('Full history differs without a request difference')
    return None


def bank_comparison(a, b, side, state):
    ca, cb = a['banks'][state], b['banks'][state]
    own, opponent = ca[side]-cb[side], ca[1-side]-cb[1-side]
    return dict(state_index=state, observation_equal=a['physical'][state] == b['physical'][state],
        candidate=dict(own_bank=ca[side], opponent_bank=ca[1-side], margin=ca[side]-ca[1-side]),
        reference=dict(own_bank=cb[side], opponent_bank=cb[1-side], margin=cb[side]-cb[1-side]),
        delta=dict(own_bank=own, opponent_bank=opponent, margin=own-opponent))


def finish_pair(pair, a, b):
    step, choice = pair['first_request_difference_step'], pair['first_choice_difference']
    pair['terminal'] = bank_comparison(a, b, pair['side'], 719)
    pair['horizons'] = {}
    pair['first_sampled_observation_difference_state'] = None
    if step is not None:
        for horizon in (1, 24):
            pair['horizons'][str(horizon)] = bank_comparison(a, b, pair['side'], min(step+horizon, 719))
        sampled = sorted({min(step+1, 719), min(step+24, 719), 719})
        pair['first_sampled_observation_difference_state'] = next(
            (i for i in sampled if a['physical'][i] != b['physical'][i]), None)
        pair['request_components_differ'] = {k: a['components'][step][k] != b['components'][step][k]
                                              for k in ('farmer', 'hands', 'market')}
    else:
        pair['request_components_differ'] = None
    unknown = pair['unalignable_common_turns'] > 0 or (step is not None and
                pair['first_request_turn_choice_linkage'] is None)
    if unknown:
        category = 'unknown'
    elif choice is None:
        category = 'no_task_disagreement'
    elif step is None:
        category = 'task_disagreement_same_request'
    elif pair['first_sampled_observation_difference_state'] is None:
        category = 'request_difference_no_recorded_observation_effect'
    elif pair['terminal']['delta']['margin'] == 0:
        category = 'observation_difference_zero_terminal_margin_gap'
    else:
        category = 'observation_difference_nonzero_terminal_margin_gap'
    pair['partition'] = category
    pair['identifiable_chain_or_null_chain'] = category != 'unknown'
    pair['completion_observability'] = 'unknown_in_saved_schema'
    pair['duration_mechanism_supported'] = False
    pair['effect_causation'] = 'unknown; observed full-policy trajectory association only'
    pair.pop('common_prefix_active')
    return pair


def descriptor():
    return dict(selected_ops=Counter(), emitted_request_kind=Counter(),
        adjacent_selected_identity_changes=0, identity_changes_after_move_request=0,
        trace_turns=0, previous={}, previous_day=None)


def aggregate(rows):
    primary = [r for r in rows if r['reference'] == 'teacher_task']
    identified = [r for r in primary if r['identifiable_chain_or_null_chain']]
    negatives = [r for r in primary if r['terminal']['delta']['margin'] < 0]
    groups = []
    for reference in ('teacher_task', 'baseline'):
        selected = [r for r in rows if r['reference'] == reference]
        for dimension in ('all', 'opponent', 'seed'):
            values = [None] if dimension == 'all' else list(dict.fromkeys(r[dimension] for r in selected))
            for value in values:
                subset = selected if dimension == 'all' else [r for r in selected if r[dimension] == value]
                groups.append(dict(reference=reference, dimension=dimension, value=value, pairs=len(subset),
                    seats=sorted(set(r['side'] for r in subset)),
                    partitions={k: sum(r['partition'] == k for r in subset) for k in PARTITIONS},
                    terminal_delta_sum={k: sum(r['terminal']['delta'][k] for r in subset)
                                        for k in ('own_bank', 'opponent_bank', 'margin')}))
    coverage = len(identified) >= 24 and len(set(r['seed'] for r in identified)) >= 6
    return dict(pairs=len(rows), primary_pairs=len(primary), groups=groups,
        screen=dict(identifiable_primary_pairs=len(identified), identifiable_primary_seeds=len(set(r['seed'] for r in identified)),
            coverage_passed=coverage, duration_supported_pairs=0, duration_unknown_pairs=len(primary),
            duration_supported_seeds=0, duration_supported_seats=[],
            negative_terminal_margin_gap_sum=sum(-r['terminal']['delta']['margin'] for r in negatives),
            supported_loss_association_coverage=0., duration_passed=False,
            passed=False, interpretation='Resource-allocation screen, not a significance test'),
        status='evidence_insufficient', new_training_authorized=False,
        conclusion='Saved observations can establish trajectory differences; missing completion evidence prevents the duration-option question.',
        causal_interpretation='No regret labels, causal return shares, counterfactual outcomes or fresh holdout claim')


def run(out):
    from .experiment import PREVIOUS, read, write_new
    started, cpu = time.monotonic(), time.process_time()
    manifest = read(Path(out)/'input-manifest.json')
    games, pairs = manifest['games'], manifest['pairs']
    identities = [g[k]['path'] for g in games for k in ('replay', 'trace')]
    if len(games) != 96 or len(pairs) != 32 or len(set(identities)) != 192:
        raise ValueError('Incomplete fixed input manifest')
    meter, rows, game_descriptors = Meter(), [], []
    try:
        for number, spec in enumerate(pairs):
            selected = {g['arm']: g for g in games
                        if (g['seed'], g['opponent'], g['side']) == (spec['seed'], spec['opponent'], spec['side'])}
            if set(selected) != set(ARMS):
                raise ValueError('Incomplete triad')
            compact = {}
            for arm in ARMS:
                game = selected[arm]
                compact[arm] = compact_replay(read_json(PREVIOUS/game['replay']['path'], game['replay'], meter), game)
            current = {}
            for reference in ('teacher_task', 'baseline'):
                first = first_request_difference(compact['candidate'], compact[reference], spec['side'])
                if reference == 'baseline' and first != spec['first_candidate_baseline_request_difference']:
                    raise ValueError('Accepted C-S first request step mismatch')
                current[reference] = dict(seed=spec['seed'], opponent=spec['opponent'], side=spec['side'],
                    reference=reference, first_request_difference_step=first, first_choice_difference=None,
                    first_request_turn_choice_linkage=None,
                    matched_context_calls=0, equal_context_choice_disagreement_turns=0,
                    disagreement_same_complete_request_turns=0, disagreement_different_complete_request_turns=0,
                    unalignable_common_turns=0, common_prefix_turns=0, common_prefix_active=True,
                    first_unalignable_reason=None)
            streams = {arm: trace_lines(PREVIOUS/selected[arm]['trace']['path'], selected[arm]['trace'], meter) for arm in ARMS}
            descriptors = {arm: descriptor() for arm in ARMS}
            try:
                for step in range(719):
                    traces = {}
                    for arm in ARMS:
                        try:
                            traces[arm] = next(streams[arm])
                        except StopIteration as exc:
                            raise ValueError('Trace ended before 719 turns') from exc
                        validate_trace(traces[arm], compact[arm], step, descriptors[arm])
                    for reference, pair in current.items():
                        a, b = compact['candidate'], compact[reference]
                        if not pair['common_prefix_active']:
                            continue
                        if a['physical'][step] != b['physical'][step]:
                            pair['common_prefix_active'] = False
                            if step == 0:
                                pair['unalignable_common_turns'] += 1
                                pair['first_unalignable_reason'] = 'initial_observation_difference'
                            continue
                        pair['common_prefix_turns'] += 1
                        comparison = compare_callbacks(traces['candidate']['records'], traces[reference]['records'])
                        pair['matched_context_calls'] += comparison['matched_context_calls']
                        if comparison['context_stop']:
                            pair['unalignable_common_turns'] += 1
                            pair['first_unalignable_reason'] = pair['first_unalignable_reason'] or comparison['context_stop']
                        if comparison['choice'] is not None:
                            if step == pair['first_request_difference_step']:
                                pair['first_request_turn_choice_linkage'] = dict(step=step, **comparison['choice'])
                            pair['equal_context_choice_disagreement_turns'] += 1
                            if pair['first_choice_difference'] is None:
                                pair['first_choice_difference'] = dict(step=step, **comparison['choice'])
                            same = a['requests'][step][spec['side']] == b['requests'][step][spec['side']]
                            pair['disagreement_same_complete_request_turns' if same else 'disagreement_different_complete_request_turns'] += 1
                for arm in ARMS:
                    try:
                        next(streams[arm])
                    except StopIteration:
                        pass
                    else:
                        raise ValueError('Trace contains extra turns')
            finally:
                for stream in streams.values():
                    stream.close()
            for arm in ARMS:
                desc = descriptors[arm]
                desc.pop('previous'); desc.pop('previous_day')
                game_descriptors.append(dict(game=selected[arm]['key'], seed=spec['seed'],
                    opponent=spec['opponent'], side=spec['side'], arm=arm, **desc,
                    completion_observability='unknown_in_saved_schema', geometry_is_descriptive_only=True))
            for reference, pair in current.items():
                rows.append(finish_pair(pair, compact['candidate'], compact[reference]))
            del compact, traces, descriptors
            print(json.dumps(dict(completed_triads=number+1, **meter.snapshot())), flush=True)
        if len(rows) != 64 or len({(r['seed'],r['opponent'],r['side'],r['reference']) for r in rows}) != 64:
            raise ValueError('Wrong final comparison identities')
        if {r['path'] for r in meter.receipts} != set(identities) or len(meter.receipts) != 192:
            raise ValueError('Not all fixed files completed exactly once')
        result = dict(schema='task-action-impact-diagnostic-result-v1', complete=True,
            **aggregate(rows), input_games=96, triads=32, raw_files=192, terminal_checks=96,
            accepted_first_request_checks=32, raw_reads=meter.snapshot(),
            analysis_wall_seconds=time.monotonic()-started, analysis_cpu_seconds=time.process_time()-cpu,
            gpu_seconds=0, neural_forwards=0, official_transitions=0, training_updates=0, new_games=0,
            decompressed_copies_on_disk=0)
        write_new(Path(out)/'paired-rows.json', rows)
        write_new(Path(out)/'game-descriptors.json', game_descriptors)
        write_new(Path(out)/'raw-read-receipts.json', meter.receipts)
        write_new(Path(out)/'analysis-result.json', result)
        return result
    except BaseException as exc:
        write_new(Path(out)/'analysis-failure.json', dict(error_type=type(exc).__name__,
            message=str(exc), **meter.snapshot(), completed_comparisons=len(rows),
            analysis_wall_seconds=time.monotonic()-started, analysis_cpu_seconds=time.process_time()-cpu,
            attempt_consumed=True, no_retry=True))
        raise
