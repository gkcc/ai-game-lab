"""Read a fixed, already accepted panel subset. No simulation or inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

GAME_ROOT = Path(__file__).resolve().parents[2]
REPO = GAME_ROOT.parent
RUNTIME_SHA256 = {
    'decision/__init__.py': '292c33c078490e76cf6576d8922f527acf23d59a01b8f523f7a1b595cbe15d0e',
    'decision/api.py': '3ef4025bde61d240b298cbe4d31291f25021baaf5b396f5540bbfcdda4231917',
    'decision/semantics.py': '9ac62d0df7adca84c0feff805b2979bb45f9f9f4c2064d1e2b5c48efd3e779a5',
    'learning/task_bc_error_diagnostics_v1/__init__.py': '7fe73d92a6f435034e2de4fc239e202b38b383508f2c426bd389dd7565f4a381',
    'learning/task_bc_error_diagnostics_v1/shadow.py': '095f6a896c259f840af4c59adc922cc9782fe69aafd3b0df7ea04df9160c3251',
    'learning/task_action_impact_diagnostic_v1/__init__.py': '2049e94f1fe2891136fc09817309da1b3bc054bba42b23bd6f5cf06ac176921d',
    'learning/task_action_impact_diagnostic_v1/analysis.py': 'f0263611bd3bddc1f9028938288a0a1ec4406ebbc3d27df2ec17772bbb581499',
    'learning/task_prefix_diagnostic_v1/__init__.py': 'b390682ccf45f9d8abe27f10827aa6a87b16b20bec06019a83142283995ff2ff',
    'learning/task_prefix_diagnostic_v1/pairing.py': '7c4341f78c5f2c2758d83b62b0388e3b0b98a97518a2ec3acdf9467881132b18',
    'learning/task_single_intervention_v1/trusted_imports.py': '329d3b6415e3fac1cabea5d27571db3510281cbabd9c5118c55cec9aa43b115a',
}


def verify_runtime():
    for name, expected in RUNTIME_SHA256.items():
        if hashlib.sha256((GAME_ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen diagnostic dependency changed')


def bind_runtime():
    global analysis, PairComparison
    verify_runtime()
    guard_path = GAME_ROOT / 'learning/task_single_intervention_v1/trusted_imports.py'
    guard_raw = guard_path.read_bytes()
    if hashlib.sha256(guard_raw).hexdigest() != RUNTIME_SHA256['learning/task_single_intervention_v1/trusted_imports.py']:
        raise ValueError('Import guard changed')
    guard = types.ModuleType('frozen_diagnostic_import_guard')
    exec(compile(guard_raw, 'frozen-diagnostic-import-guard', 'exec', dont_inherit=True), guard.__dict__)
    guard.bind(GAME_ROOT, check_public_prefix=True)
    from learning.task_action_impact_diagnostic_v1 import analysis as frozen_analysis
    from learning.task_prefix_diagnostic_v1.pairing import PairComparison as frozen_comparison
    analysis, PairComparison = frozen_analysis, frozen_comparison

LEDGER_SHA256 = '481d4bc82e3e705c0c4750d2dd8f984098e65eef334231d1225169c43bd459ba'
REGISTRATION_SHA256 = '293478899d30f47830add190e699b3077d0466806d2276cee0a4828b822bb245'
FREEZE_SHA256 = '249d72b0770bb97cb4a1cc66fc14586dd674c778185f6aebf8c50c73fad9cb4d'
PROJECTOR_SHA256 = '6409fe65dfa5f332edc52ee1d2edccf8694a08f27be20de9a3d460b0ef88a9b3'
RESULT_COMMIT = '3a3e3172b0403a171a38f6e78e37d07f387be2cb'
REGISTRATION = REPO/'review/evidence/task-teacher-merged-eval-v3-formal-20261030/registration.json'
LEDGER = REPO/'review/evidence/task-teacher-merged-eval-v3-formal-result-20261030/alias-ledger.json'
OPPONENTS = ('round2',)
SEED_ALIASES = (('S02', 1), ('S03', 2))
ARMS = ('merged', 'unweighted', 'teacher_task')
COMPRESSED_LIMIT, DECODED_LIMIT, WALL_SECONDS = 512*2**20, 16*2**30, 900


def sha(data):
    return hashlib.sha256(data).hexdigest()


def lf_sha(data):
    return sha(data.replace(b'\r\n', b'\n'))


def read_small(path):
    with path.open('rb') as stream:
        raw = stream.read(4*2**20+1)
    if len(raw) > 4*2**20:
        raise ValueError('Metadata size limit')
    return raw


def load_small(path, expected=None, normalized=False):
    raw = read_small(path)
    if expected is not None and (lf_sha(raw) if normalized else sha(raw)) != expected:
        raise ValueError('Metadata identity mismatch')
    return json.loads(raw.decode('utf-8-sig'))


def safe_file(root, *parts):
    path = root.joinpath(*parts)
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Panel input missing or outside panel')
    return path


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def check_deadline(started, clock=time.monotonic):
    """Sample elapsed wall time; this is not an OS-enforced hard deadline."""
    require(clock()-started <= WALL_SECONDS, 'Sampled analysis deadline exceeded')


def equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def verify_completion(panel, status_path):
    """Screen existing program receipts; they are not an independent replay."""
    status = load_small(status_path)
    require(status.get('state') == 'accepted_and_published' and status.get('result_commit') == RESULT_COMMIT
            and status.get('remote_main_verified') is True and status.get('process_exit_code') == 0,
            'Local panel completion status not accepted')
    data = read_small(REPO/'tools/panel_v3_alias_ledger.py')
    require(lf_sha(data) == PROJECTOR_SHA256, 'Completion checker identity differs')
    projector = types.ModuleType('historical_panel_completion_checker')
    exec(compile(data, 'historical-panel-completion-checker', 'exec', dont_inherit=True), projector.__dict__)
    files = (('games-result.json',), ('independent-runner-result.json',),
             ('independent-replay-verification.json',), ('results', 'coordinator-complete.json'),
             ('replays', 'launcher-result.json'))
    values = [load_small(safe_file(panel, *parts)) for parts in files]
    projector.require_non_economic_completion(*values, status['process_exit_code'])
    return files


def check_replay_seed(replay, row):
    require(isinstance(replay, dict) and isinstance(replay.get('info'), dict)
            and type(row.get('seed')) is int and type(replay['info'].get('seed')) is int
            and replay['info']['seed'] == row['seed'], 'Replay seed differs from accepted row')


def check_trace_identity(trace, row):
    require(isinstance(trace, dict) and 'checkpoint_sha256' in trace and 'task_checkpoint_sha256' in trace
            and trace['checkpoint_sha256'] == row['crop_checkpoint_sha256']
            and trace['task_checkpoint_sha256'] == row['task_checkpoint_sha256'], 'Trace model identity differs')


def check_counts(counts, limits, policy):
    fields = {'task_forwards', 'task_rows', 'crop_forwards', 'crop_rows',
              'gate_evaluations', 'controller_calls', 'teacher_proposals'}
    require(isinstance(counts, dict) and set(counts) == fields
            and all(type(v) is int and v >= 0 for v in counts.values())
            and counts['controller_calls'] == 719
            and counts['controller_calls'] <= limits['controller_calls']
            and counts['task_forwards'] <= limits['task_forwards']
            and counts['task_rows'] >= counts['task_forwards']
            and counts['crop_forwards'] <= limits['crop_forwards']
            and counts['crop_rows'] >= counts['crop_forwards']
            and counts['task_rows'] + counts['crop_rows'] <= limits['neural_input_rows']
            and counts['gate_evaluations'] <= limits['gate_evaluations']
            and counts['teacher_proposals'] <= limits['teacher_proposals'], 'Recorded model budget invalid')
    require((policy == 'teacher_task' and counts['task_forwards'] == counts['task_rows'] == 0)
            or (policy != 'teacher_task' and counts['task_forwards'] > 0), 'Recorded task model invalid')


def count_sources(arm, records, counts):
    require(arm in ARMS and isinstance(records, list), 'Invalid task selection records')
    expected = 'teacher' if arm == 'teacher_task' else 'callback'
    require(all(isinstance(r, dict) and r.get('source') == expected for r in records),
            'Task selection source does not match policy')
    counts['selection_records'] += len(records)
    counts[expected+'_records'] += len(records)
    counts['zero_selection_turns'] += not records
    counts['zero_callback_turns'] += not any(r['source'] == 'callback' for r in records)


def check_selected(selected):
    expected = {(alias, opponent, side, policy) for alias, _ in SEED_ALIASES
                for opponent in OPPONENTS for side in (0, 1) for policy in ARMS}
    require(set(selected) == expected, 'Selected roles or games differ')
    role_seeds = {}
    for (alias, opponent, side, policy), item in selected.items():
        row = item['row']
        require(row.get('seed_alias') == alias and row.get('opponent') == opponent
                and type(row.get('side')) is int and row['side'] == side
                and row.get('policy') == policy and type(row.get('seed')) is int,
                'Selected row identity differs')
        if alias in role_seeds:
            require(role_seeds[alias] == row['seed'], 'Inconsistent seed role')
        role_seeds[alias] = row['seed']
    require(len(set(role_seeds.values())) == len(SEED_ALIASES), 'Seed roles overlap')


def checked_rows(panel, status_path):
    """Bind 12 historical rows to their existing local acceptance receipts.

    The receipts are program records, not independent proof of raw gameplay.
    Raw hashes and private plan values stay inside this process and local run.
    """
    ledger = load_small(LEDGER, LEDGER_SHA256, normalized=True)
    cfg = load_small(REGISTRATION, REGISTRATION_SHA256, normalized=True)
    verify_completion(panel, status_path)
    load_small(safe_file(panel, 'registration.json'), REGISTRATION_SHA256, normalized=True)
    load_small(safe_file(panel, 'freeze.json'), FREEZE_SHA256, normalized=True)
    seeds = load_small(safe_file(panel, 'private-seeds.local.json'), cfg['seed_document_sha256'])
    require(isinstance(seeds, dict) and isinstance(seeds.get('seeds'), list)
            and len(seeds['seeds']) == 4 and all(type(v) is int and 0 <= v < 2**32 for v in seeds['seeds'])
            and len(set(seeds['seeds'])) == 4, 'Private seed document invalid')
    game_result = load_small(safe_file(panel, 'games-result.json'))
    require(game_result.get('passed') is True and game_result.get('complete_games') == 72
            and game_result.get('live_official_transitions') == 72*719
            and game_result.get('primary_replay_transitions') == 72*719, 'Panel not accepted')
    audit_rows = load_small(safe_file(panel, 'game-audit-ledger.json'), game_result['audit_ledger_sha256'])
    require(isinstance(ledger, list) and len(ledger) == 72 and isinstance(audit_rows, list)
            and len(audit_rows) == 72, 'Panel ledger incomplete')
    audits = {r['game']: r for r in audit_rows}
    require(len(audits) == 72, 'Duplicate audit rows')
    selected = {}
    for alias, seed_index in SEED_ALIASES:
        selected_seed = seeds['seeds'][seed_index]
        for opponent in OPPONENTS:
            for side in (0, 1):
                for policy in ARMS:
                    key = f'holdout-{alias}-{opponent}-{side}-{policy}'
                    found = [r for r in ledger if r.get('game') == key]
                    require(len(found) == 1, 'Missing public alias row')
                    published, audit_row = found[0], audits[key]
                    acceptance_path = safe_file(panel, 'job-acceptance', key+'.json')
                    acc = load_small(acceptance_path, audit_row['acceptance_receipt_sha256'])
                    require(acc.get('passed') is True and acc.get('game') == key
                            and acc.get('retry_allowed') is False
                            and acc.get('freeze_sha256') == sha(read_small(safe_file(panel, 'freeze.json'))),
                            'Game acceptance missing')
                    evidence = acc['evidence']
                    supervision_doc = load_small(safe_file(panel, 'job-supervision', key+'.json'),
                                                 acc['supervision_sha256'])
                    supervision = supervision_doc['supervision']
                    require(evidence.get('game') == key and evidence.get('official_transitions') == 719
                            and evidence.get('replay_sha256') == audit_row['replay_sha256']
                            and evidence.get('trace_sha256') == audit_row['trace_sha256']
                            and acc['supervision_sha256'] == audit_row['supervision_receipt_sha256']
                            and supervision_doc.get('game') == key
                            and supervision_doc.get('freeze_sha256') == acc['freeze_sha256']
                            and supervision_doc.get('job_input_sha256') == acc['job_input_sha256']
                            and supervision_doc.get('retry_allowed') is False
                            and equal(supervision, evidence.get('job_supervision'))
                            and equal(supervision, audit_row.get('job_supervision'))
                            and all(supervision.get(k) is True for k in ('passed', 'assigned', 'resumed',
                                                                        'launcher_reaped', 'job_members_zero'))
                            and supervision.get('exit_code') == 0
                            and all(supervision.get(k) is None for k in ('error_kind', 'cleanup_error_kind', 'stop_reason')),
                            'Acceptance mismatch')
                    plan = load_small(safe_file(panel, 'job-inputs', key+'.json'), acc['job_input_sha256'])
                    receipt_paths = {name: safe_file(panel, 'job-results', key+'.'+name+'.json')
                                     for name in ('row', 'audit', 'complete', 'storage')}
                    receipts = {name: load_small(path, evidence['receipt_sha256'][name])
                                for name, path in receipt_paths.items()}
                    row, audit, complete, storage = (receipts[n] for n in ('row', 'audit', 'complete', 'storage'))
                    require(all(k in row and equal(row[k], v) for k, v in published.items())
                            and all(k in row and equal(row[k], v) for k, v in plan.items()), 'Public row or plan mismatch')
                    require(row.get('game') == key and row.get('seed_alias') == alias
                            and type(row.get('seed')) is int and row['seed'] == selected_seed
                            and row.get('opponent') == opponent and row.get('side') == side
                            and row.get('policy') == policy and row.get('states') == 720
                            and row.get('action_turns') == 719 and row.get('status') == ['DONE', 'DONE']
                            and row.get('agent_errors') == 0, 'Unexpected historical row')
                    task = None if policy == 'teacher_task' else cfg[policy+'_checkpoint_sha256']
                    require(row.get('task_checkpoint_sha256') == task
                            and evidence.get('task_checkpoint_sha256') == task
                            and row.get('crop_checkpoint_sha256') == cfg['crop_checkpoint_sha256']
                            and row.get('engine_sha256') == cfg['engine_sha256']
                            and row.get('rival_sha256') == cfg['source_sha256'][opponent+'.py']
                            and sha(read_small(safe_file(panel, 'agents', key+'.py'))) == row.get('bot_sha256')
                            and sha(read_small(safe_file(panel, 'sources', opponent+'.py'))) == row.get('rival_sha256'),
                            'Recorded model, engine or wrapper identity mismatch')
                    official = audit.get('official', {})
                    counts = audit.get('primary_model_counts')
                    check_counts(counts, cfg['game_limits'], policy)
                    output = storage.get('output', {})
                    require(audit.get('passed') is True and audit.get('game') == key and audit.get('errors') == 0
                            and audit.get('actor_actions_reproduced') == 719
                            and audit.get('rival_actions_reproduced') == 719
                            and audit.get('trace_turns_reproduced') == 719
                            and audit.get('replay_sha256') == row['replay_gzip_sha256'] == evidence['replay_sha256']
                            and audit.get('trace_sha256') == row['trace_sha256'] == evidence['trace_sha256']
                            and official.get('passed') is True and official.get('official_transitions_reexecuted') == 719
                            and official.get('engine_sha256') == cfg['engine_sha256']
                            and official.get('both_public_and_private_states_match_each_step') is True
                            and official.get('inventory_order_checked_each_step') is True
                            and official.get('semantic_sha256') == row['semantic_replay_sha256']
                            and official.get('semantic_sha256_v2') == row['semantic_replay_sha256_v2']
                            and official.get('final_rewards') == row['rewards']
                            and complete.get('passed') is True and complete.get('game') == key
                            and complete.get('row_sha256') == evidence['receipt_sha256']['row']
                            and complete.get('certificate_sha256') == evidence['receipt_sha256']['audit']
                            and storage.get('passed') is True and storage.get('game') == key
                            and storage.get('error_kind') is None
                            and output.get('limit_bytes') == cfg['storage']['game_write_bytes']
                            and type(output.get('used_bytes')) is int and 0 <= output['used_bytes'] <= output['limit_bytes']
                            and output.get('remaining_bytes') == output['limit_bytes']-output['used_bytes']
                            and output.get('refused') == 0
                            and counts == audit.get('live_model_counts'), 'Historical replay receipt mismatch')
                    replay = safe_file(panel, 'games', key, 'replay.json.gz')
                    trace = safe_file(panel, 'traces', key+'.jsonl.gz')
                    selected[(alias, opponent, side, policy)] = dict(row=row, counts=counts,
                        replay=dict(path=f'games/{key}/replay.json.gz', size=replay.stat().st_size,
                                    sha256=row['replay_gzip_sha256']),
                        trace=dict(path=f'traces/{key}.jsonl.gz', size=trace.stat().st_size,
                                   sha256=row['trace_sha256']))
    check_selected(selected)
    return selected


def input_snapshot(panel, status_path, selected):
    """Freeze current expected identities privately, after previous acceptance.

    The snapshot is post-hoc and cannot prove what was present when the
    historical panel was first accepted. The raw files are checked when read.
    """
    metadata = ('games-result.json', 'game-audit-ledger.json', 'independent-runner-result.json',
                'independent-replay-verification.json', 'results/coordinator-complete.json',
                'replays/launcher-result.json')
    hashes = {name: sha(read_small(safe_file(panel, *name.split('/')))) for name in metadata}
    hashes['local-completion-status'] = sha(read_small(status_path))
    inputs = {game['row']['game']: {'replay_sha256': game['replay']['sha256'],
                                    'trace_sha256': game['trace']['sha256']}
              for game in selected.values()}
    require(len(inputs) == 12, 'Input snapshot incomplete')
    return dict(schema='task-prefix-private-input-snapshot-v2', result_commit=RESULT_COMMIT,
                metadata_sha256=hashes, expected_inputs=inputs)


def write_result_new(output, result):
    """Publish success only after a complete, closed, verified local write."""
    payload = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    pending, final = output/'result.pending.json', output/'result.json'
    with pending.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(pending.read_bytes() == payload and not final.exists(), 'Result write verification failed')
    os.replace(pending, final)


def public_pair(alias, comparison):
    require(alias in ('S02', 'S03'), 'Unexpected seed role')
    value = comparison.finish()
    allowed = {'side', 'opponent', 'reference', 'first_branch_kind', 'first_branch_step',
        'branch_attribution', 'first_actor_request_components_differ', 'eligible_turns',
        'actor_observation_or_configuration_mismatch_turns', 'zero_both_selection_record_turns',
        'input_difference_turns', 'selection_record_count_difference_turns', 'equal_recorded_choices_turns',
        'choice_difference_turns', 'choice_difference_same_actor_request_turns',
        'choice_difference_different_actor_request_turns', 'choice_difference_same_request_same_task_multiset_turns',
        'matched_context_selection_records', 'teacher_label_consistent_selection_records',
        'teacher_label_disagreement_selection_records', 'candidate_teacher_agreement_selection_records',
        'reference_teacher_agreement_selection_records', 'first_choice_difference',
        'choice_difference_at_branch', 'post_branch_windows'}
    require(set(value) == allowed, 'Pair projection fields differ')
    require(type(value['side']) is int and value['side'] in (0, 1)
            and value['opponent'] == 'round2' and value['reference'] in ('unweighted', 'teacher_task')
            and value['first_branch_kind'] in ('actor', 'opponent', 'simultaneous', 'state', 'terminal_state', 'none')
            and value['branch_attribution'] == 'unknown', 'Pair projection identity differs')
    branch_step = value['first_branch_step']
    kind = value['first_branch_kind']
    require((kind == 'none' and branch_step is None)
            or (kind == 'terminal_state' and type(branch_step) is int and branch_step == 719)
            or (kind in ('actor', 'opponent', 'simultaneous', 'state')
                and type(branch_step) is int and 0 <= branch_step < 719), 'Invalid projected branch step')
    count_fields = {name for name in allowed if name.endswith('_turns') or name.endswith('_records')}
    require(all(type(value[name]) is int and value[name] >= 0 for name in count_fields),
            'Invalid projected count')
    require(value['choice_difference_at_branch'] is None or type(value['choice_difference_at_branch']) is bool,
            'Invalid projected choice indicator')
    choice = value.get('first_choice_difference')
    if choice is not None:
        require(set(choice) == {'step', 'selection_record_index', 'candidate_op', 'reference_op',
            'same_unit', 'same_target', 'candidate_request_kind', 'reference_request_kind',
            'selected_unit_request_equal', 'complete_actor_request_equal'}, 'Choice projection fields differ')
        require(type(choice['step']) is int and 0 <= choice['step'] < 719
                and type(choice['selection_record_index']) is int and choice['selection_record_index'] >= 0
                and all(choice[name] in ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
                        for name in ('candidate_op', 'reference_op'))
                and all(choice[name] in ('move', 'work', 'other')
                        for name in ('candidate_request_kind', 'reference_request_kind'))
                and all(type(choice[name]) is bool for name in ('same_unit', 'same_target',
                        'selected_unit_request_equal', 'complete_actor_request_equal')),
                'Invalid projected choice')
    components = value.get('first_actor_request_components_differ')
    if components is not None:
        require(set(components) == {'farmer', 'hands', 'market'}
                and all(type(item) is bool for item in components.values()), 'Action projection fields differ')
    value['seed_alias'] = alias
    windows = value.get('post_branch_windows')
    if windows is not None:
        require(kind == 'actor' and set(windows) == {'1', '24'}, 'Window projection fields differ')
        for horizon, window in windows.items():
            require(set(window) == {'state_index', 'semantic_physical_state_equal',
                'own_bank_delta_changed', 'opponent_bank_delta_changed', 'margin_delta_changed'},
                'Window projection fields differ')
            require(type(branch_step) is int and type(window['state_index']) is int
                    and window['state_index'] == min(branch_step + int(horizon), 719)
                    and all(type(window[name]) is bool for name in ('semantic_physical_state_equal',
                        'own_bank_delta_changed', 'opponent_bank_delta_changed', 'margin_delta_changed')),
                    'Invalid projected window')
            window['actions_elapsed'] = window['state_index'] - branch_step
            window['observation_equal'] = window.pop('semantic_physical_state_equal')
    return value


def preserve_shed_order(compact, replay):
    """Extend frozen semantic digests with both players' ordered shed items."""
    frames = replay['steps']
    require(len(frames) == 720 and len(compact['full']) == len(compact['physical']) == 720,
            'Invalid replay length for shed ordering')
    for step, frame in enumerate(frames):
        require(len(frame) == 2, 'Invalid player count for shed ordering')
        sheds = [player['observation']['private']['shed'] for player in frame]
        require(all(type(shed) is dict for shed in sheds), 'Invalid shed state')
        ordered = analysis.ordered_hash(sheds)
        compact['full'][step] = (compact['full'][step], ordered)
        compact['physical'][step] = (compact['physical'][step], ordered)
    return compact


def run(panel, status_path, manifest, progress):
    started = time.monotonic()
    check_deadline(started)
    selected = checked_rows(panel, status_path)
    require(equal(manifest, input_snapshot(panel, status_path, selected)), 'Private input snapshot changed')
    check_deadline(started)
    meter = analysis.Meter(compressed_limit=COMPRESSED_LIMIT, decoded_limit=DECODED_LIMIT)
    pairs = []
    selection_counts = {arm: dict(selection_records=0, callback_records=0, teacher_records=0,
                                  zero_selection_turns=0, zero_callback_turns=0) for arm in ARMS}
    for alias, _ in SEED_ALIASES:
        for opponent in OPPONENTS:
            for side in (0, 1):
                games = {arm: selected[(alias, opponent, side, arm)] for arm in ARMS}
                compact = {}
                for arm, game in games.items():
                    check_deadline(started)
                    replay = panel/game['replay']['path']
                    value = analysis.read_json(replay, game['replay'], meter)
                    check_replay_seed(value, game['row'])
                    compact[arm] = analysis.compact_replay(value, dict(side=side, expected=game['row']))
                    compact[arm] = preserve_shed_order(compact[arm], value)
                    del value
                    check_deadline(started)
                comparisons = {ref: PairComparison(compact['merged'], compact[ref], side, opponent, ref)
                               for ref in ('unweighted', 'teacher_task')}
                streams = {arm: analysis.trace_lines(panel/game['trace']['path'], game['trace'], meter)
                           for arm, game in games.items()}
                descriptions = {arm: analysis.descriptor() for arm in ARMS}
                recorded = {arm: 0 for arm in ARMS}
                try:
                    for step in range(719):
                        check_deadline(started)
                        records = {}
                        for arm in ARMS:
                            check_deadline(started)
                            try:
                                trace = next(streams[arm])
                            except StopIteration as exc:
                                raise ValueError('Trace ended early') from exc
                            check_deadline(started)
                            check_trace_identity(trace, games[arm]['row'])
                            records[arm] = analysis.validate_trace(trace, compact[arm], step, descriptions[arm])
                            recorded[arm] += len(records[arm])
                            count_sources(arm, records[arm], selection_counts[arm])
                        for ref, comparison in comparisons.items():
                            comparison.observe_turn(step, records['merged'], records[ref])
                    for arm in ARMS:
                        check_deadline(started)
                        try:
                            next(streams[arm])
                        except StopIteration:
                            pass
                        else:
                            raise ValueError('Trace has extra turns')
                        if arm == 'teacher_task':
                            require(games[arm]['counts'].get('task_forwards') == 0, 'Unexpected teacher task model')
                        else:
                            require(recorded[arm] == games[arm]['counts'].get('task_forwards'),
                                    'Recorded task calls differ from reconstructed count')
                finally:
                    for stream in streams.values():
                        stream.close()
                pairs.extend(public_pair(alias, comparison) for comparison in comparisons.values())
                check_deadline(started)
                progress['triads_completed'] += 1
                print(json.dumps({'triads_completed': progress['triads_completed']}), flush=True)
    require(len(pairs) == 8 and len(meter.receipts) == 24 and len(meter.opened) == 24,
            'Input or pairing coverage incomplete')
    check_deadline(started)
    return dict(schema='task-prefix-diagnostic-v2', passed=True, seed_aliases=['S02', 'S03'],
        input_games=12, paired_comparisons=8, pairs=pairs, selection_counts=selection_counts,
        raw_reads=meter.snapshot(), wall_deadline='sampled, not OS-enforced',
        decoded_budget_is_memory_limit=False,
        seconds=time.monotonic()-started, new_games=0, official_transitions=0,
        neural_forwards=0, training_updates=0, gpu_operations=0,
        interpretation='Exploratory historical prefix associations; no causal, significance or holdout claim.')


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('Invalid command line')


def main():
    parser = QuietParser(description=__doc__)
    parser.add_argument('--panel', required=True, type=Path, help='Local accepted panel directory')
    parser.add_argument('--status', required=True, type=Path, help='Local panel completion status')
    parser.add_argument('--manifest', required=True, type=Path, help='Private input snapshot')
    parser.add_argument('--prepare-only', action='store_true', help='Create snapshot without decoding games')
    parser.add_argument('--output', type=Path, help='New local output directory')
    args = parser.parse_args()
    bind_runtime()
    panel = args.panel.resolve(strict=True)
    status_path = args.status.resolve(strict=True)
    manifest_path = args.manifest.resolve()
    if manifest_path.is_relative_to(panel) or manifest_path.is_relative_to(REPO.resolve()):
        raise ValueError('Private manifest must be outside the source panel and public repository')
    if args.prepare_only:
        require(args.output is None, 'Preparation does not create an analysis output')
        selected = checked_rows(panel, status_path)
        snapshot = input_snapshot(panel, status_path, selected)
        verify_runtime()
        with manifest_path.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        print(json.dumps({'passed': True, 'input_games': 12, 'raw_games_decoded': 0}), flush=True)
        return
    require(args.output is not None, 'Output directory required')
    manifest = load_small(manifest_path)
    output = args.output.resolve()
    if output.is_relative_to(panel) or output.is_relative_to(REPO.resolve()):
        raise ValueError('Output must be outside the source panel and public repository')
    output.mkdir(parents=False, exist_ok=False)
    progress = dict(triads_completed=0)
    try:
        result = run(panel, status_path, manifest, progress)
        verify_runtime()
        write_result_new(output, result)
    except BaseException as exc:
        with (output/'failure.json').open('x', encoding='utf-8', newline='\n') as f:
            json.dump(dict(passed=False, error_type=type(exc).__name__, **progress), f, indent=2)
            f.write('\n')
        with (output/'failure.local.log').open('x', encoding='utf-8', newline='\n') as f:
            f.write(traceback.format_exc())
        print(json.dumps({'passed': False, 'error_type': type(exc).__name__, **progress}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({'passed': True, 'input_games': 12, 'paired_comparisons': 8}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        print(json.dumps({'passed': False, 'error_type': type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
