"""Bind task origins to decision inputs, then save private numeric arrays."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from learning.task_action_impact_diagnostic_v1.analysis import Meter, compact_replay, read_json, trace_lines
from learning.task_action_observation_v1.analysis import raw_path
from learning.task_completion_evidence_v1.labels import EpisodeTracker
from learning.task_completion_evidence_v1.pipeline import EventJoin, selections_from_trace, validate_rows
from . import data, numeric
from .experiment import ACTION, OUT, PARENT, deadline_check, read, ready, sha, write_new


def labels_file(path, expected, maximum):
    path = Path(path)
    if path.stat().st_size > maximum or sha(path) != expected:
        raise ValueError('Episode label identity or size mismatch')
    rows, decoded = [], 0
    with gzip.open(path, 'rb') as stream:
        while True:
            line = stream.readline(min(1024**2, maximum-decoded+1))
            if not line:
                break
            decoded += len(line)
            if decoded > maximum or len(line) >= 1024**2 or not line.endswith(b'\n') or len(rows) >= 16000:
                raise ValueError('Episode labels exceed decoded or row budget')
            rows.append(json.loads(line))
    if sha(path) != expected:
        raise ValueError('Episode label file changed while reading')
    return rows


def summarize(groups):
    counts, per_seed, per_seed_status = Counter(), Counter(), Counter()
    for g in groups:
        counts[(g['candidate_ops'][g['chosen']], g['status'])] += 1
        per_seed[str(g['seed'])] += 1
        per_seed_status[(str(g['seed']), g['status'])] += 1
    return dict(groups=len(groups), candidates=sum(len(g['candidates']) for g in groups),
                completed_observed=sum(g['status']=='completed_observed' for g in groups),
                unknown=sum(g['status']=='unknown' for g in groups),
                counts=[dict(op=k[0], status=k[1], count=v) for k, v in sorted(counts.items())],
                per_seed=dict(sorted(per_seed.items())),
                per_seed_status=[dict(seed=k[0], status=k[1], count=v) for k, v in sorted(per_seed_status.items())])


def checked_turns(path, identity, meter):
    for turn in trace_lines(path, identity, meter):
        deadline_check()
        yield turn


def train(out=OUT):
    out = Path(out).resolve()
    cfg = ready(out)
    if (out/'train-data-started.json').exists():
        raise ValueError('Training-data attempt already consumed')
    write_new(out/'train-data-started.json', dict(attempt=1, retry_allowed=False,
        freeze_sha256=sha(out/'freeze.json'), input_role='eight prior development games'))
    limits = cfg['data_limits']
    meter = Meter(limits['train_compressed_bytes'], limits['train_decoded_bytes'])
    schema, groups, completed = read(out/'schema.json'), [], []
    try:
        for game in cfg['source_groups']:
            deadline_check()
            path = (PARENT/game['labels_path']).resolve()
            if not path.is_relative_to(PARENT/'private/games'):
                raise ValueError('Historical labels escaped their source')
            rows = labels_file(path, game['labels_sha256'], limits['episode_file_bytes'])
            if len(rows) != game['labels_rows']:
                raise ValueError('Historical label count changed')
            turns = checked_turns(raw_path(ACTION, game['trace'], 'trace'), game['trace'], meter)
            try:
                actual_schema, batch = data.collect_game(game, rows, turns, schema)
            finally:
                turns.close()
            if actual_schema != schema or len(batch) != len(rows):
                raise ValueError('Historical episode join incomplete')
            groups.extend(batch)
            if len(groups) > limits['max_groups_per_split'] or sum(len(g['candidates']) for g in groups) > limits['max_candidates_per_split']:
                raise ValueError('Training group budget exceeded')
            completed.append(dict(game=game['key'], arm=game['arm'], groups=len(batch), labels_sha256=game['labels_sha256'],
                                  trace_sha256=game['trace']['sha256']))
        if len(completed) != 8 or len(meter.receipts) != 8:
            raise ValueError('Incomplete development inputs')
        arrays = numeric.pack(groups, limits['max_groups_per_split'], limits['max_candidates_per_split'])
        deadline_check()
        private = out/'private'
        private.mkdir(exist_ok=True)
        target = private/'train.npz'
        numeric.write_new(target, arrays)
        deadline_check()
        result = dict(passed=True, source_role='previously observed development only',
                      source_policies='two baseline, four candidate, two teacher_task; actual selected tasks, not pure teacher imitation',
                      games=completed, **summarize(groups), raw_reads=meter.snapshot(),
                      array_sha256=sha(target), array_bytes=target.stat().st_size,
                      feature_boundary='shared and candidate arrays only; seed and outcome are metadata',
                      supervision='first actual chosen; only completed_observed enters gradients',
                      actor_trace_provenance='prior frozen actor reconstruction plus replay and trace hashes',
                      raw_labels_and_features_published=False)
        write_new(out/'train-data-result.json', result)
        return result
    except BaseException as exc:
        write_new(out/'train-data-failure.local.json', dict(error_kind=type(exc).__name__,
            completed_games=len(completed), raw_reads=meter.snapshot(), retry_allowed=False))
        raise


def current_game(out, row):
    key = row['game']
    folder = out/'job-results'
    certificate, marker = read(folder/(key+'.audit.json')), read(folder/(key+'.complete.json'))
    if (not marker['passed'] or marker['game'] != key or marker['row_sha256'] != sha(folder/(key+'.row.json'))
            or marker['certificate_sha256'] != sha(folder/(key+'.audit.json'))
            or not row['passed'] or row['states'] != 720 or row['action_turns'] != 719
            or row['status'] != ['DONE','DONE'] or row['agent_errors'] != 0
            or not certificate['passed'] or certificate['errors'] != 0
            or certificate['actor_actions_reproduced'] != 719
            or certificate['rival_actions_reproduced'] != 719 or certificate['trace_turns_reproduced'] != 719):
        raise ValueError('Official job or actor reconstruction is incomplete')
    replay_path = out/'games'/key/'replay.json.gz'
    trace_path = out/'traces'/(key+'.jsonl.gz')
    if (certificate['replay_sha256'] != row['replay_gzip_sha256']
            or certificate['trace_sha256'] != row['trace_sha256']):
        raise ValueError('Official source identity mismatch')
    expected = {k: row[k] for k in ('banks','bank','opponent_bank','margin','net_profit',
                                     'opponent_net_profit','rewards','status','score')}
    return dict(key=key, seed=row['seed'], side=row['side'], opponent=row['opponent'],
        arm=row['policy'], expected=expected,
        replay=dict(path=replay_path.relative_to(out).as_posix(), size=replay_path.stat().st_size,
                    sha256=row['replay_gzip_sha256']),
        trace=dict(path=trace_path.relative_to(out).as_posix(), size=trace_path.stat().st_size,
                   sha256=row['trace_sha256'])), certificate


def test(out=OUT):
    """One separate label replay per fixed game; trace has two bounded passes."""
    from decision.full_audit import reexecute
    from .campaign import verify_model_seal
    out = Path(out).resolve()
    cfg = ready(out)
    games_result = read(out/'games-result.json')
    verify_model_seal(out)
    if (not games_result['passed'] or not games_result['all_owned_processes_reaped']
            or games_result['game_ledger_sha256'] != sha(out/'game-ledger.json')
            or games_result['audit_ledger_sha256'] != sha(out/'game-audit-ledger.json')):
        raise ValueError('Frozen model and complete test collection required')
    if (out/'test-data-started.json').exists():
        raise ValueError('Test-data attempt already consumed')
    write_new(out/'test-data-started.json', dict(attempt=1, retry_allowed=False,
        freeze_sha256=sha(out/'freeze.json'), label_replays=8,
        trace_decode_passes=2, replay_decode_passes=1))
    limits = cfg['data_limits']
    primary = Meter(limits['test_compressed_bytes']//2, limits['test_decoded_bytes']//2)
    joining = Meter(limits['test_compressed_bytes']//2, limits['test_decoded_bytes']//2)
    schema, groups, completed, transitions = read(out/'schema.json'), [], [], 0
    try:
        rows = read(out/'game-ledger.json')
        expected = {(s, op, side) for s in cfg['test_seeds'] for op in cfg['opponents'] for side in cfg['seats']}
        actual = {(row['seed'], row['opponent'], row['side']) for row in rows}
        if (len(rows) != 8 or len({row['game'] for row in rows}) != 8 or actual != expected
                or any(row['policy'] != 'teacher_task' or row['split'] != 'holdout' for row in rows)):
            raise ValueError('Invalid registered test population')
        label_folder = out/'private/labels'
        label_folder.mkdir(parents=True, exist_ok=False)
        for row in rows:
            deadline_check()
            game, certificate = current_game(out, row)
            replay_path = raw_path(out, game['replay'], 'replay')
            trace_path = raw_path(out, game['trace'], 'trace')
            replay = read_json(replay_path, game['replay'], primary)
            compact = compact_replay(replay, game)
            stream = trace_lines(trace_path, game['trace'], primary)
            try:
                selections = selections_from_trace(stream, compact, deadline_check)
            finally:
                stream.close()
            join = EventJoin(EpisodeTracker(game['key'], game['side']), selections,
                             compact['days'], game['side'], deadline_check)
            audit = reexecute(replay, trace=True, label_sink=join.sink)
            transitions += audit['official_transitions_reexecuted']
            if (not audit['passed'] or audit['official_transitions_reexecuted'] != 719
                    or audit['engine_sha256'] != cfg['engine_sha256']
                    or audit['semantic_sha256'] != row['semantic_replay_sha256']
                    or audit['semantic_sha256_v2'] != row['semantic_replay_sha256_v2']
                    or audit['semantic_sha256_v2'] != certificate['official']['semantic_sha256_v2']):
                raise ValueError('Separate official label replay mismatch')
            labels = validate_rows(join.finish(), game, sum(map(len, selections)))
            payload = b''.join(json.dumps(r, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':')).encode('utf-8')+b'\n' for r in labels)
            compressed = gzip.compress(payload, mtime=0)
            if len(compressed) > limits['episode_file_bytes']:
                raise ValueError('Private label output exceeds budget')
            label_path = label_folder/(game['key']+'.jsonl.gz')
            with label_path.open('xb') as f:
                f.write(compressed)
            turns = checked_turns(trace_path, game['trace'], joining)
            try:
                actual_schema, batch = data.collect_game(game, labels, turns, schema)
            finally:
                turns.close()
            if actual_schema != schema or len(batch) != len(labels):
                raise ValueError('Test episode join incomplete')
            groups.extend(batch)
            if len(groups) > limits['max_groups_per_split'] or sum(len(g['candidates']) for g in groups) > limits['max_candidates_per_split']:
                raise ValueError('Test group budget exceeded')
            completed.append(dict(game=game['key'], seed=game['seed'], opponent=game['opponent'],
                side=game['side'], groups=len(batch), labels_sha256=hashlib.sha256(compressed).hexdigest(),
                replay_sha256=game['replay']['sha256'], trace_sha256=game['trace']['sha256'],
                semantic_sha256_v2=audit['semantic_sha256_v2'], official_transitions=719))
            del replay, compact, selections, join, labels, payload, compressed, batch
        if len(completed) != 8 or transitions != 8*719 or len(primary.receipts) != 16 or len(joining.receipts) != 8:
            raise ValueError('Incomplete registered test-data extraction')
        arrays = numeric.pack(groups, limits['max_groups_per_split'], limits['max_candidates_per_split'])
        deadline_check()
        target = out/'private/test.npz'
        numeric.write_new(target, arrays)
        deadline_check()
        write_new(out/'test-label-ledger.json', completed)
        result = dict(passed=True, games=8, official_transitions=transitions,
                      **summarize(groups), primary_raw_reads=primary.snapshot(),
                      second_trace_pass=joining.snapshot(), array_sha256=sha(target),
                      array_bytes=target.stat().st_size,
                      label_ledger_sha256=sha(out/'test-label-ledger.json'),
                      model_seal_sha256=sha(out/'model-seal.json'), games_result_sha256=sha(out/'games-result.json'),
                      split='prospective two-seed offline evaluation, not candidate-versus-opponent matches',
                      outcome_is_metric_only=True, private_rows_and_features_published=False)
        write_new(out/'test-data-result.json', result)
        return result
    except BaseException as exc:
        write_new(out/'test-data-failure.local.json', dict(error_kind=type(exc).__name__,
            completed_games=len(completed), completed_official_transitions=transitions,
            primary_raw_reads=primary.snapshot(), second_trace_pass=joining.snapshot(), retry_allowed=False))
        raise
