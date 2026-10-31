"""Bounded, private sampling from the original task-choice demonstrations.

The caller supplies bytes and binary streams; this module never opens or writes
files, runs an agent, or starts the official engine. All returned identities,
observations, prefixes and requests are private and must stay local. The
acceptance digest must be anchored to previously authenticated evidence, not
calculated from the same unverified file at the time of this call.

``sample_game`` consumes all three gzip streams to EOF, even after finding the
six selected groups. Seventy-two is a selection limit, not a decoded-row limit.
"""

from copy import deepcopy
from dataclasses import dataclass
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import PurePosixPath
import re

from decision.api import observe
from learning.bc_crop.agent import TEACHER_SHA
from learning.bc_crop.experiment import ENGINE_COMMIT, ENGINE_SHA
from learning.task_bc_v1.cache import decode
from learning.task_choice_v1.dataset import digest
from learning.task_choice_v1.features import SCHEMA_VERSION, schema_hash


if not __debug__:
    raise RuntimeError('Dataset validation requires assertions enabled')

OPPONENTS = ('market', 'round2')
ALL_OPPONENTS = ('v1', 'market', 'round2')
WINDOWS = ((0, 239), (240, 479), (480, 718))
PER_WINDOW = 2
MIN_CANDIDATES, MAX_CANDIDATES = 2, 256
MAX_COMPRESSED_FILE = 128 * 1024 * 1024
MAX_REPLAY_DECODED = 128 * 1024 * 1024
MAX_LINE = 8 * 1024 * 1024
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_GAME = re.compile(r'[A-Za-z0-9_-]+\Z')
_INPUT_FIELDS = frozenset(('schema_version', 'schema_sha256', 'feature_names',
    'shared_feature_names', 'shared_x', 'candidate_scope', 'observation',
    'configuration', 'planned', 'candidates', 'inventory_item_order',
    'shed_item_order'))
_INDEX_FIELDS = frozenset(('game', 'seed', 'opponent', 'side', 'step',
                           'decision', 'split', 'input_sha256'))
_PLANNED_FIELDS = frozenset(('seeds', 'needed_seeds', 'supply', 'projected_shed',
                             'unit_actions', 'available_units', 'interpretation'))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _sha_text(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def _seed(value):
    return type(value) is int and 0 <= value < 2**32


def _loads(raw):
    """Reject duplicate keys and non-finite JSON values at every level."""
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError('Nonfinite JSON value')

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Invalid JSON') from exc


def _document(raw):
    _require(type(raw) is bytes and bool(raw), 'Expected nonempty evidence bytes')
    value = _loads(raw.decode('utf-8-sig'))
    _require(type(value) is dict, 'Evidence must be an object')
    return value


@dataclass(frozen=True, repr=False)
class FileBinding:
    path: str
    sha256: str
    rows: int


@dataclass(frozen=True, repr=False)
class GameBinding:
    game: str
    seed: int
    opponent: str
    side: int
    source_split: str
    split: str
    replay_sha256: str
    schema_sha256: str
    files: tuple

    def file(self, kind):
        return dict(self.files)[kind]


def bind_sources(*, registration_raw, manifest_raw, ledger_raw, review_raw, schema_raw,
                 acceptance_raw, expected_acceptance_sha256, excluded_seeds):
    """Authenticate the old evidence chain and create exactly 12 private plans.

    ``expected_acceptance_sha256`` must originate in a separately authenticated
    old Git snapshot or previously fixed trusted digest. The caller must also
    authenticate the two excluded *actual* seeds against the old round2 panel;
    aliases are deliberately neither accepted nor returned. Evidence identity
    is not a third-party timestamp or proof of provenance beyond that anchor.
    """
    _require(type(acceptance_raw) is bytes and _sha_text(expected_acceptance_sha256)
             and _sha(acceptance_raw) == expected_acceptance_sha256,
             'Acceptance anchor differs')
    cfg, manifest, ledger, review, source_schema, acceptance = map(_document,
        (registration_raw, manifest_raw, ledger_raw, review_raw, schema_raw, acceptance_raw))
    evidence = acceptance.get('evidence_sha256')
    _require(type(evidence) is dict, 'Missing acceptance evidence mapping')
    for name, raw in (('registration.json', registration_raw),
                      ('dataset-manifest.json', manifest_raw),
                      ('reconstruction-ledger.json', ledger_raw),
                      ('independent-review.json', review_raw), ('schema.json', schema_raw)):
        _require(evidence.get(name) == _sha(raw), 'Old evidence binding differs')
    _require(acceptance.get('stage_execution_passed') is True
             and acceptance.get('structural_coverage_passed') is True
             and acceptance.get('teacher_mode_full_action_equivalence_passed') is True
             and acceptance.get('original_demonstrations_reconstructed') == 60
             and acceptance.get('all_dataset_rows_independently_consumed') is True
             and acceptance.get('dataset_files') == 180,
             'Old acceptance did not pass')
    _require(review.get('passed') is True and review.get('all_dataset_files_hashed') == 180
             and review.get('all_dataset_rows_aligned') == manifest.get('examples')
             and review.get('ledger_sha256') == _sha(ledger_raw),
             'Independent old review differs')
    _require(cfg.get('experiment') == 'joint-task-choice-interface-v1'
             and cfg.get('demonstrations') == 60 and cfg.get('training_runs') == 0
             and cfg.get('new_complete_games') == 0
             and cfg.get('engine_commit') == ENGINE_COMMIT
             and cfg.get('engine_sha256') == ENGINE_SHA
             and cfg.get('teacher_sha256') == TEACHER_SHA,
             'Old task-choice registration differs')
    _require(manifest.get('registration_sha256') == _sha(registration_raw)
             and manifest.get('reconstruction_ledger_sha256') == _sha(ledger_raw)
             and manifest.get('teacher_sha256') == TEACHER_SHA
             and manifest.get('schema_version') == SCHEMA_VERSION
             and _sha_text(manifest.get('schema_sha256'))
             and manifest.get('source_ledgers') == cfg.get('source_ledgers')
             and manifest.get('training_runs') == 0 and manifest.get('new_games') == 0,
             'Dataset manifest differs')
    names, shared_names = source_schema.get('feature_names'), source_schema.get('shared_feature_names')
    _require(type(names) is list and len(names) == 98
             and type(shared_names) is list and len(shared_names) == 275
             and all(type(name) is str for name in names + shared_names)
             and len(set(names)) == 98 and len(set(shared_names)) == 275
             and source_schema.get('schema_version') == SCHEMA_VERSION
             and source_schema.get('schema_sha256') == manifest['schema_sha256']
             and schema_hash(names, shared_names) == manifest['schema_sha256']
             and source_schema.get('shared_dimension') == 275
             and source_schema.get('candidate_dimension') == 98
             and source_schema.get('labels_separate') is True
             and source_schema.get('metadata_seed_game_separate') is True
             and source_schema.get('inventory_order_explicit') is True,
             'Original feature schema differs')
    _require(ledger.get('passed') is True and ledger.get('games') == 60
             and ledger.get('games_dispatched') == 60 and ledger.get('failures') == []
             and ledger.get('new_games') == 0 and ledger.get('training_runs') == 0
             and ledger.get('official_existing_replay_transitions') == 60 * 719,
             'Reconstruction ledger did not pass')

    train, validation = cfg.get('train_seeds'), cfg.get('validation_seeds')
    _require(type(train) is list and len(train) == 8
             and type(validation) is list and len(validation) == 2
             and all(_seed(s) for s in train + validation)
             and len(set(train + validation)) == 10, 'Original seed roles differ')
    _require(type(excluded_seeds) in (list, tuple) and len(excluded_seeds) == 2
             and all(_seed(s) for s in excluded_seeds) and len(set(excluded_seeds)) == 2,
             'Expected two authenticated old-panel seeds')
    forbidden = set(excluded_seeds)
    chosen_train = [s for s in train if s not in forbidden][:2]
    _require(len(chosen_train) == 2, 'Not enough disjoint original training seeds')
    chosen_dev = [s for s in validation if s not in forbidden and s not in chosen_train][:1]
    _require(len(chosen_dev) == 1, 'No disjoint original validation seed')

    rows, files = ledger.get('rows'), manifest.get('files')
    _require(type(rows) is list and len(rows) == 60 and type(files) is list and len(files) == 60,
             'Expected the complete original 60-game population')
    file_by_game = {}
    for item in files:
        _require(type(item) is dict and type(item.get('game')) is str
                 and _GAME.fullmatch(item['game']) is not None and item['game'] not in file_by_game,
                 'Invalid or duplicate dataset game')
        file_by_game[item['game']] = item
    coordinates, bindings, total = set(), {}, 0
    for row in rows:
        _require(type(row) is dict and type(row.get('game')) is str
                 and _GAME.fullmatch(row['game']) is not None, 'Invalid reconstruction game')
        game, seed, opponent, side = row['game'], row.get('seed'), row.get('opponent'), row.get('side')
        _require(_seed(seed) and seed in train + validation and opponent in ALL_OPPONENTS
                 and type(side) is int and side in (0, 1), 'Invalid reconstruction role')
        source_split = 'train' if seed in train else 'validation'
        coord = (seed, opponent, side)
        _require(coord not in coordinates and game not in bindings and row.get('split') == source_split,
                 'Duplicate or misassigned game')
        coordinates.add(coord)
        audit = row.get('audit')
        _require(row.get('passed') is True and row.get('reconstructed_actor_actions') == 719
                 and row.get('original_teacher_actions_checked') == 719
                 and row.get('reproduced_rival_actions') == 719
                 and _sha_text(row.get('replay_gzip_sha256'))
                 and row.get('schema_version') == manifest['schema_version']
                 and row.get('schema_sha256') == manifest['schema_sha256']
                 and type(audit) is dict and audit.get('passed') is True
                 and audit.get('official_transitions_reexecuted') == 719
                 and audit.get('both_public_and_private_states_match_each_step') is True
                 and audit.get('inventory_order_checked_each_step') is True
                 and audit.get('engine_sha256') == ENGINE_SHA,
                 'Original reconstruction evidence differs')
        item = file_by_game.get(game)
        _require(type(item) is dict and item.get('split') == source_split
                 and item.get('files') == row.get('files') and type(row.get('files')) is dict
                 and set(row['files']) == {'inputs', 'labels', 'index'}, 'Game file binding differs')
        count = row.get('decisions')
        _require(type(count) is int and count > 0, 'Invalid decision count')
        stream_bindings = []
        for kind in ('inputs', 'labels', 'index'):
            spec = row['files'][kind]
            expected_path = f'data/{source_split}/{game}.{kind}.jsonl.gz'
            _require(type(spec) is dict and set(spec) == {'path', 'sha256', 'rows'}
                     and spec.get('path') == expected_path and _sha_text(spec.get('sha256'))
                     and type(spec.get('rows')) is int and spec['rows'] == count,
                     'Invalid stream binding')
            path = PurePosixPath(spec['path'])
            _require(not path.is_absolute() and '..' not in path.parts and '\\' not in spec['path'],
                     'Unsafe dataset path')
            stream_bindings.append((kind, FileBinding(spec['path'], spec['sha256'], count)))
        total += count
        split = 'train' if seed in chosen_train else 'dev'
        bindings[game] = GameBinding(game, seed, opponent, side, source_split, split,
                                     row['replay_gzip_sha256'], manifest['schema_sha256'],
                                     tuple(stream_bindings))
    expected = {(s, op, side) for s in train + validation for op in ALL_OPPONENTS for side in (0, 1)}
    _require(coordinates == expected and set(bindings) == set(file_by_game)
             and total == manifest.get('examples') == acceptance.get('task_examples'),
             'Original population or example count differs')
    by_coordinate = {(v.seed, v.opponent, v.side): v for v in bindings.values()}
    plan = tuple(by_coordinate[(seed, opponent, side)]
                 for seed in chosen_train + chosen_dev for opponent in OPPONENTS for side in (0, 1))
    _require(len(plan) == 12 and sum(item.split == 'train' for item in plan) == 8,
             'Selected plan differs')
    return plan


class ReadBudget:
    """Shared aggregate byte accounting and an injectable wall/memory guard."""
    def __init__(self, *, compressed_limit, decoded_limit, check):
        _require(type(compressed_limit) is int and compressed_limit > 0
                 and type(decoded_limit) is int and decoded_limit > 0 and callable(check),
                 'Invalid read budget')
        self.compressed_limit, self.decoded_limit, self.check = compressed_limit, decoded_limit, check
        self.compressed_bytes = self.decoded_bytes = 0
        self.stream_lines = {'inputs': 0, 'labels': 0, 'index': 0}
        self.replay_json_attempts = self.started_groups = self.parsed_groups = 0

    def guard(self):
        _require(self.check() is True, 'Read guard stopped')

    def decoded(self, value):
        self.decoded_bytes += len(value)
        _require(self.decoded_bytes <= self.decoded_limit, 'Decoded byte limit exceeded')
        self.guard()


class _HashingStream:
    def __init__(self, source, budget):
        self.source, self.budget = source, budget
        self.hasher, self.count = hashlib.sha256(), 0

    def read(self, size=-1):
        self.budget.guard()
        if size < 0:
            parts = []
            while True:
                chunk = self.read(65536)
                if not chunk:
                    return b''.join(parts)
                parts.append(chunk)
        limit = min(MAX_COMPRESSED_FILE - self.count, self.budget.compressed_limit - self.budget.compressed_bytes)
        data = self.source.read(min(size, max(0, limit) + 1))
        _require(type(data) is bytes, 'Expected a binary stream')
        self.count += len(data)
        self.budget.compressed_bytes += len(data)
        self.hasher.update(data)
        _require(self.count <= MAX_COMPRESSED_FILE
                 and self.budget.compressed_bytes <= self.budget.compressed_limit,
                 'Compressed byte limit exceeded')
        return data


def _json_line(stream, budget, kind):
    line = stream.readline(MAX_LINE + 1)
    if line:
        budget.stream_lines[kind] += 1
    budget.decoded(line)
    _require(len(line) <= MAX_LINE and (not line or line.endswith(b'\n')),
             'Truncated or oversized data line')
    return line


def _checked_gzip(source, expected, budget):
    _require(_sha_text(expected), 'Invalid source digest')
    hashing = _HashingStream(source, budget)
    return hashing, gzip.GzipFile(fileobj=hashing, mode='rb')


def _restore_mapping(mapping, order):
    _require(type(mapping) is dict and type(order) is list
             and all(type(k) is str for k in order) and len(order) == len(set(order))
             and set(order) == set(mapping), 'Inventory ordering metadata differs')
    return {key: deepcopy(mapping[key]) for key in order}


def restore_input_order(inputs):
    """Restore precisely the inventory/shed order lost to sort_keys storage."""
    _require(type(inputs) is dict and set(inputs) == _INPUT_FIELDS, 'Unexpected model input fields')
    result = deepcopy(inputs)
    obs = result.get('observation')
    _require(type(obs) is dict and type(obs.get('private')) is dict, 'Missing own private view')
    private = obs['private']
    inventories, orders = private.get('inventories'), result['inventory_item_order']
    _require(type(inventories) is list and type(orders) is list and len(inventories) == len(orders),
             'Inventory count differs')
    private['inventories'] = [_restore_mapping(inv, order) for inv, order in zip(inventories, orders)]
    private['shed'] = _restore_mapping(private.get('shed'), result['shed_item_order'])
    return result


def _replay(source, binding, budget):
    hashing, zipped = _checked_gzip(source, binding.replay_sha256, budget)
    parts, size = [], 0
    try:
        while True:
            budget.guard()
            chunk = zipped.read(min(1024 * 1024, MAX_REPLAY_DECODED - size + 1))
            budget.decoded(chunk)
            if not chunk:
                break
            size += len(chunk)
            _require(size <= MAX_REPLAY_DECODED, 'Replay decoded limit exceeded')
            parts.append(chunk)
    finally:
        zipped.close()
    _require(hashing.hasher.hexdigest() == binding.replay_sha256, 'Replay digest differs')
    budget.replay_json_attempts += 1
    replay = _loads(b''.join(parts))
    _require(type(replay) is dict and type(replay.get('info')) is dict
             and type(replay['info'].get('seed')) is int and replay['info']['seed'] == binding.seed
             and type(replay.get('configuration')) is dict and type(replay.get('steps')) is list
             and len(replay['steps']) == 720, 'Replay identity differs')
    for step, frame in enumerate(replay['steps']):
        _require(type(frame) is list and len(frame) == 2 and all(type(s) is dict for s in frame)
                 and type(frame[0].get('observation')) is dict
                 and frame[0]['observation'].get('step') == step, 'Replay frame order differs')
    _require([state.get('status') for state in replay['steps'][-1]] == ['DONE', 'DONE'],
             'Replay did not finish')
    return replay


def _view(frame, configuration, side):
    """Project only the actor's private data and the shared public fields."""
    own = frame[side].get('observation')
    public = frame[0]['observation']
    _require(type(own) is dict and type(own.get('player')) is int and own['player'] == side
             and type(own.get('private')) is dict, 'Actor replay observation differs')
    raw = {'player': side, 'private': deepcopy(own['private'])}
    for name in ('step', 'farms', 'market', 'town', 'day', 'hour'):
        _require(name in public, 'Missing public replay field')
        raw[name] = deepcopy(public[name])
    return observe(raw, configuration)


def sample_game(binding, *, inputs_stream, labels_stream, index_stream, replay_stream, budget):
    """Verify one full game and return six local-only, fixed-rule samples.

    Inputs are raw gzip binary streams at their start. Each stream is consumed
    to EOF and its actual compressed digest checked against the authenticated
    metadata. The caller owns and closes the passed streams and is responsible
    for the outer 12-game uniqueness, output protection and any time limit.
    """
    _require(type(binding) is GameBinding and type(budget) is ReadBudget, 'Invalid sampling context')
    replay = _replay(replay_stream, binding, budget)
    streams = (inputs_stream, labels_stream, index_stream)
    kinds = ('inputs', 'labels', 'index')
    opened = [_checked_gzip(source, binding.file(kind).sha256, budget)
              for kind, source in zip(kinds, streams)]
    selected, buckets, previous, prefix = [], [0, 0, 0], None, []
    count, schema = 0, None
    missing = object()
    try:
        # A binary sentinel is necessary because a blank line is invalid JSON,
        # while b'' is the only EOF marker.
        lines = (iter(lambda z=zipped, k=kind: _json_line(z, budget, k), b'')
                 for (_hash, zipped), kind in zip(opened, kinds))
        for triple in zip_longest(*lines, fillvalue=missing):
            budget.started_groups += 1
            _require(all(item is not missing for item in triple), 'Unaligned stream lengths')
            inputs, label, index = map(_loads, triple)
            _require(type(inputs) is dict and type(label) is dict and type(index) is dict
                     and set(index) == _INDEX_FIELDS and set(label) == {'teacher_label', 'index'}
                     and type(inputs.get('planned')) is dict and set(inputs['planned']) == _PLANNED_FIELDS,
                     'Unexpected record fields')
            step, decision = index['step'], index['decision']
            _require(type(step) is int and 0 <= step < 719 and type(decision) is int and decision >= 0,
                     'Invalid decision position')
            _require(type(index['seed']) is int and index['seed'] == binding.seed
                     and type(index['side']) is int and index['side'] == binding.side
                     and index['opponent'] == binding.opponent and index['game'] == binding.game
                     and index['split'] == binding.source_split, 'Record identity differs')
            current = (step, decision)
            _require(previous is None or current > previous, 'Decision sequence is not ordered')
            if previous is None or previous[0] != step:
                _require(decision == 0, 'Decision sequence does not start at zero')
                prefix = []
            else:
                _require(decision == previous[1] + 1, 'Decision sequence has a gap')
            previous = current
            _require(index.get('input_sha256') == digest(inputs), 'Input digest differs')
            try:
                schema, _shared, _local, _label_index, _ops, _ids = decode(inputs, label, index,
                    game=binding.game, split=binding.source_split, expected_seed=binding.seed,
                    expected_side=binding.side, expected_opponent=binding.opponent, schema=schema)
            except (AssertionError, KeyError, TypeError, ValueError, IndexError) as exc:
                raise ValueError('Aligned input or label validation failed') from exc
            _require(schema['schema_sha256'] == binding.schema_sha256, 'Dataset schema differs')
            restored = restore_input_order(inputs)
            frame = replay['steps'][step]
            view = _view(frame, replay['configuration'], binding.side)
            saved = restored['observation']
            _require(saved == view.observation and restored['configuration'] == view.configuration
                     and list(saved['private']['shed']) == list(view.private['shed'])
                     and [list(v) for v in saved['private']['inventories']]
                         == [list(v) for v in view.private['inventories']],
                     'Saved input differs from the actor-visible replay view')
            candidates = inputs['candidates']
            window = next(i for i, (lo, hi) in enumerate(WINDOWS) if lo <= step <= hi)
            if buckets[window] < PER_WINDOW and MIN_CANDIDATES <= len(candidates) <= MAX_CANDIDATES:
                following = replay['steps'][step + 1][binding.side].get('action')
                _require(type(following) is dict and set(following) == {'farmer', 'hands', 'market'}
                         and all(type(following[k]) is list for k in following), 'Missing recorded actor request')
                selected.append(dict(split=binding.split, step=step, decision=decision,
                    inputs=restored, label=deepcopy(label), teacher_prefix=tuple(prefix),
                    recorded_actor_request=deepcopy(following)))
                buckets[window] += 1
            prefix.append(label['teacher_label'])
            count += 1
            budget.parsed_groups += 1
            budget.guard()
        _require(count == binding.file('inputs').rows
                 and all(hashing.hasher.hexdigest() == binding.file(kind).sha256
                         for (hashing, _zipped), kind in zip(opened, kinds)),
                 'Stream row count or digest differs')
        _require(buckets == [PER_WINDOW] * len(WINDOWS) and len(selected) == 6,
                 'One or more fixed windows have too few qualifying groups')
        return selected
    finally:
        for _hash, zipped in opened:
            zipped.close()
