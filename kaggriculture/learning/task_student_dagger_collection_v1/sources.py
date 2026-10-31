"""Authenticate completed DAgger student games without reading reserved seeds.

All replay states, identities, model paths, and original game identifiers stay
private. Only the 24 entries in the fixed, published completion certificate
are addressable. The caller must authenticate this module and the new stage's
registration before importing it, own resource/process limits and never expose
replay data or original coordinates to a model or public log.

This module never opens the historical registration, seeds.json, private seed
inventory, or old preparation functions that parse unrun holdout roles.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import MappingProxyType

from learning.task_student_continuation_v1 import sources as old
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA, GAME_LIMITS


RESULT_COMMIT = '7141729e00913a2418d06abc9aac87b8f9a83a74'
FRESH_COMMIT = '40f3f393819bbdc62af1007877cd5fbdf3a9a42f'
ROOT_SHA256 = '98142750672977a458655f3fb953789e607108cd352ba3be117d647f0f402279'
RESULT_PREFIX = 'review/evidence/task-dagger-once-v1-20261027/'
FRESH_REGISTRATION = 'review/evidence/task-pair-fresh-multi-v1-20261031/registration.json'
STAGE_NAME = 'task-dagger-once-v1-20261027'
TRAIN_ROLES = ('T01', 'T02', 'T03', 'T04')
DEV_ROLES = ('V01', 'V02')
ROLES = TRAIN_ROLES + DEV_ROLES
OPPONENTS = ('market', 'round2')
FILES = frozenset(('input', 'row', 'audit', 'completion', 'result', 'replay', 'trace', 'wrapper'))
MAX_SMALL = 8 * 1024 * 1024
MAX_LINE = 8 * 1024 * 1024
MAX_DECODED = 512 * 1024 * 1024
GAME_NAME = re.compile(r'[A-Za-z0-9_-]{1,200}\Z')
RUNTIME_COMPONENTS = (
    'learning/task_dagger_once_v1/policy.py', 'learning/task_choice_v1/agent.py',
    'learning/task_choice_v1/features.py', 'learning/task_bc_v1/model.py',
    'learning/bc_crop/agent.py', 'learning/bc_crop/features.py',
    'learning/task_request_gate_v1/policy.py', 'decision/api.py',
    'bots/teacher_fertilizer_terminal_v2.py',
)


class SourceError(ValueError):
    """A source or its public evidence chain did not authenticate."""


def _require(value, message):
    if not value:
        raise SourceError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    _require(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
             'invalid source digest')
    return value


def _json(raw, maximum=MAX_SMALL):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, 'invalid source document size')
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate source document key')
            result[key] = value
        return result
    def nonfinite(_):
        raise SourceError('nonfinite source document value')
    try:
        value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError):
        raise SourceError('invalid source document') from None
    _require(type(value) is dict, 'source document must be an object')
    return value


def _git_blob(commit, relative, guard):
    guard()
    _require((commit == RESULT_COMMIT and relative in {
        RESULT_PREFIX + 'root-result-acceptance.json',
        RESULT_PREFIX + 'independent-result-review.json',
        RESULT_PREFIX + 'independent-collection-review.json',
        RESULT_PREFIX + 'freeze.json',
    }) or (commit == FRESH_COMMIT and relative == FRESH_REGISTRATION), 'source blob not allowed')
    try:
        value = subprocess.run(['git', '-C', str(old.fresh.REPO), 'show', commit + ':' + relative],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=30, check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        raise SourceError('source blob unavailable') from None
    guard()
    _require(value.returncode == 0 and 0 < len(value.stdout) <= MAX_SMALL, 'source blob unavailable')
    return value.stdout


def _small(root, path, limit, guard):
    return old._small(root, path, limit, guard)


def _assert_blob(root, path, expected, guard):
    raw = _small(root, path, MAX_SMALL, guard)
    _require(_sha(raw) == _digest(expected), 'source file digest differs')
    return raw


@dataclass(frozen=True, repr=False)
class _Entry:
    game: str
    seed: int
    split: str
    opponent: str
    side: int
    hashes: object

    def __repr__(self):
        return '<PrivateCompletedGameEntry>'


class PrivateContext:
    def __init__(self, entries, task_path, task_sha, crop_path, crop_sha):
        self._entries = MappingProxyType(dict(entries))
        self._task_path, self._task_sha = task_path, task_sha
        self._crop_path, self._crop_sha = crop_path, crop_sha
        self._active = False
        # A visit is consumed when authentication starts, even on failure.
        self._visits = {}
        self._completed_scans = {}

    def __repr__(self):
        return '<PrivateDaggerSourceContext>'

    def model_normalization(self):
        """Load the authenticated task network without opening a game or inference."""
        from learning.task_bc_v1.model import TaskNetwork
        network = TaskNetwork(self._task_path, self._task_sha)
        return network.mean.copy(), network.scale.copy(), deepcopy(network.metadata)


def _review_entries(review):
    _require(review.get('passed') is True and review.get('exact_registered_population_complete') is True,
             'completion review did not pass')
    records = review.get('reviewed_games')
    _require(type(records) is list and len(records) == 24, 'completed population differs')
    seed_order = {'train': [], 'dev': []}
    checked, names = [], set()
    for record in records:
        _require(type(record) is dict and record.get('passed') is True and record.get('policy') == 'baseline',
                 'invalid completed game')
        split, seed = record.get('split'), record.get('seed')
        game, opponent, side = record.get('game'), record.get('opponent'), record.get('side')
        _require(split in ('train', 'dev') and type(seed) is int and 0 <= seed < 2**32
                 and type(game) is str and GAME_NAME.fullmatch(game) and game not in names
                 and opponent in OPPONENTS and type(side) is int and side in (0, 1),
                 'invalid completed game coordinate')
        hashes = record.get('input_sha256')
        _require(type(hashes) is dict and set(hashes) == FILES, 'incomplete source evidence')
        hashes = {key: _digest(hashes[key]) for key in FILES}
        if seed not in seed_order[split]:
            seed_order[split].append(seed)
        names.add(game)
        checked.append(_Entry(game, seed, split, opponent, side, MappingProxyType(hashes)))
    _require(len(seed_order['train']) == 4 and len(seed_order['dev']) == 2
             and not set(seed_order['train']) & set(seed_order['dev']), 'completed role counts differ')
    role_map = {**dict(zip(seed_order['train'], TRAIN_ROLES)), **dict(zip(seed_order['dev'], DEV_ROLES))}
    entries = {}
    for item in checked:
        key = (role_map[item.seed], item.opponent, item.side)
        _require(key not in entries, 'duplicate completed coordinate')
        entries[key] = item
    expected = {(role, opponent, side) for role in ROLES for opponent in OPPONENTS for side in (0, 1)}
    _require(set(entries) == expected, 'completed source panel is incomplete')
    return entries


def open_context(*, guard):
    """Verify the public completion chain and the prior baseline commitment."""
    _require(callable(guard), 'resource guard required')
    guard()
    root_raw = _git_blob(RESULT_COMMIT, RESULT_PREFIX + 'root-result-acceptance.json', guard)
    _require(_sha(root_raw) == ROOT_SHA256, 'published root digest differs')
    root = _json(root_raw)
    _require(root.get('passed') is True and type(root.get('official_games')) is int
             and root['official_games'] == 24 and root.get('candidate_admission') is False,
             'historical completion root differs')
    review_raw = _git_blob(RESULT_COMMIT, RESULT_PREFIX + 'independent-result-review.json', guard)
    _require(_sha(review_raw) == _digest(root.get('independent_result_review_sha256')),
             'independent result review digest differs')
    review = _json(review_raw)
    _require(review.get('passed') is True and type(review.get('certificate_sha256')) is dict,
             'independent result review did not pass')
    collection_raw = _git_blob(RESULT_COMMIT, RESULT_PREFIX + 'independent-collection-review.json', guard)
    _require(_sha(collection_raw) == _digest(review['certificate_sha256'].get('independent-collection-review.json')),
             'independent collection review digest differs')
    entries = _review_entries(_json(collection_raw))

    freeze_raw = _git_blob(RESULT_COMMIT, RESULT_PREFIX + 'freeze.json', guard)
    _require(_sha(freeze_raw) == _digest(root.get('freeze_sha256')), 'historical freeze digest differs')
    freeze = _json(freeze_raw)
    local, public = freeze.get('runtime_files'), freeze.get('published_runtime_files')
    _require(type(local) is dict and type(public) is dict, 'historical runtime sources missing')
    for relative in RUNTIME_COMPONENTS:
        guard()
        _assert_blob(old.fresh.GAME, old.fresh.GAME / relative, local.get(relative), guard)
        _assert_blob(old.fresh.REPO, old.fresh.REPO / 'kaggriculture' / relative, public.get(relative), guard)
    _require(old._rule_sha(guard) == ENGINE_SHA, 'official game engine differs')

    fresh_reg = _json(_git_blob(FRESH_COMMIT, FRESH_REGISTRATION, guard))
    _require(fresh_reg.get('engine_commit') == ENGINE_COMMIT and fresh_reg.get('engine_sha256') == ENGINE_SHA,
             'fresh published game rules differ')
    fresh_raw = _small(old.fresh.PRIVATE_ROOT, old.fresh.PRIVATE / 'binding.local.json', 65536, guard)
    _require(_sha(fresh_raw) == _digest(fresh_reg.get('binding_commitment_sha256')),
             'fresh commitment differs')
    fresh_doc = _json(fresh_raw, 65536)
    _require(fresh_doc.get('schema') == 'task-pair-fresh-multi-private-binding-v1',
             'fresh commitment schema differs')
    market_path = old.fresh.PRIVATE_ROOT / 'task-pair-market-consumed-v1-20261031' / 'binding.local.json'
    market_raw = _small(old.fresh.PRIVATE_ROOT, market_path, 65536, guard)
    _require(_sha(market_raw) == _digest(fresh_doc.get('market_binding_sha256')),
             'baseline market binding differs')
    market = _json(market_raw, 65536)
    _require(market.get('schema') == 'task-pair-market-consumed-binding-v1'
             and type(market.get('models')) is dict and type(market['models'].get('baseline')) is dict
             and type(market.get('crop')) is dict, 'baseline model binding missing')
    task, crop = market['models']['baseline'], market['crop']
    _require(set(task) == {'path', 'sha256'} and set(crop) == {'path', 'sha256'}, 'model binding shape differs')
    task_sha, crop_sha = _digest(task['sha256']), _digest(crop['sha256'])
    task_path = old._model_file(task['path'], task_sha, guard)
    crop_path = old._model_file(crop['path'], crop_sha, guard)
    guard()
    return PrivateContext(entries, task_path, task_sha, crop_path, crop_sha)


def _verify_wrapper(raw, task_sha, crop_sha):
    try:
        tree = ast.parse(raw.decode('utf-8'))
        imports = [item for item in ast.walk(tree) if isinstance(item, ast.ImportFrom)
                   and item.module == 'learning.task_dagger_once_v1.policy'
                   and any(name.name == 'BoundedAgent' for name in item.names)]
        calls = [item for item in ast.walk(tree) if isinstance(item, ast.Call)
                 and isinstance(item.func, ast.Name) and item.func.id == 'BoundedAgent']
        _require(len(imports) == len(calls) == 1 and len(calls[0].args) == 4
                 and ast.literal_eval(calls[0].args[1]) == crop_sha
                 and ast.literal_eval(calls[0].args[3]) == task_sha,
                 'baseline controller wrapper differs')
        limits = [item.value for item in calls[0].keywords if item.arg == 'limits']
        _require(len(limits) == 1 and ast.literal_eval(limits[0]) == GAME_LIMITS,
                 'baseline controller budget differs')
    except (UnicodeError, SyntaxError, ValueError, TypeError):
        raise SourceError('baseline controller wrapper invalid') from None


def _validate_receipts(item, job, row, audit, complete, result, context):
    _require(type(row) is dict and row.get('game') == item.game and type(row.get('seed')) is int
             and row['seed'] == item.seed
             and row.get('split') == item.split and row.get('policy') == 'baseline'
             and row.get('opponent') == item.opponent and type(row.get('side')) is int
             and row['side'] == item.side and row.get('task_checkpoint_sha256') == context._task_sha
             and row.get('crop_checkpoint_sha256') == context._crop_sha and row.get('engine_sha256') == ENGINE_SHA
             and row.get('passed') is True and type(row.get('states')) is int and row['states'] == 720
             and type(row.get('action_turns')) is int and row['action_turns'] == 719
             and row.get('status') == ['DONE', 'DONE'] and type(row.get('agent_errors')) is int
             and row['agent_errors'] == 0, 'completed row differs')
    _require(row.get('replay_gzip_sha256') == item.hashes['replay']
             and row.get('trace_sha256') == item.hashes['trace']
             and row.get('bot_sha256') == item.hashes['wrapper'], 'completed file identities differ')
    for key in ('semantic_replay_sha256', 'semantic_replay_sha256_v2'):
        _digest(row.get(key))
    _require(type(job) is dict and all(type(job.get(key)) is type(row.get(key)) and job.get(key) == row.get(key)
             for key in ('game', 'seed', 'split', 'policy', 'opponent', 'side', 'parity_kind',
                         'task_checkpoint_sha256', 'bot_sha256', 'rival_sha256')),
             'completed job input differs')
    official = audit.get('official') if type(audit) is dict else None
    _require(type(audit) is dict and audit.get('game') == item.game and audit.get('passed') is True
             and type(audit.get('errors')) is int and audit['errors'] == 0
             and all(type(audit.get(k)) is int and audit[k] == 719 for k in
                     ('actor_actions_reproduced', 'rival_actions_reproduced', 'trace_turns_reproduced'))
             and audit.get('trace_sha256') == item.hashes['trace']
             and audit.get('replay_sha256') == item.hashes['replay'] and type(official) is dict
             and official.get('passed') is True and official.get('engine_sha256') == ENGINE_SHA
             and type(official.get('official_transitions_reexecuted')) is int
             and official['official_transitions_reexecuted'] == 719
             and official.get('both_public_and_private_states_match_each_step') is True
             and official.get('inventory_order_checked_each_step') is True
             and official.get('semantic_sha256') == row.get('semantic_replay_sha256')
             and official.get('semantic_sha256_v2') == row.get('semantic_replay_sha256_v2')
             and type(row.get('banks')) is list and len(row['banks']) == 2
             and official.get('final_rewards') == row['banks'] == row.get('rewards'),
             'official audit receipt differs')
    _require(type(complete) is dict and complete.get('game') == item.game and complete.get('passed') is True
             and complete.get('row_sha256') == item.hashes['row']
             and complete.get('certificate_sha256') == item.hashes['audit'], 'completion receipt differs')
    _require(type(result) is dict and result.get('passed') is True and result.get('done') is True
             and result.get('statuses') == ['DONE', 'DONE'] and type(result.get('recorded_states')) is int
             and result['recorded_states'] == 720 and type(result.get('action_turns')) is int
             and result['action_turns'] == 719 and result.get('agent_errors') == []
             and result.get('engine_matches_official_commit') is True and result.get('engine_sha256') == ENGINE_SHA
             and result.get('upstream_commit') == ENGINE_COMMIT
             and type(result.get('seed')) is int and result['seed'] == item.seed
             and type(result.get('bot_side')) is int and result['bot_side'] == item.side
             and result.get('final_money') == row['banks'] == result.get('rewards'),
             'completed official result differs')
    counts = audit.get('primary_model_counts')
    _require(type(counts) is dict and counts == audit.get('live_model_counts')
             and type(counts.get('task_forwards')) is int and counts['task_forwards'] > 0
             and type(counts.get('controller_calls')) is int and counts['controller_calls'] == 719,
             'completed model counts differ')


class _TraceReader:
    """Allow exactly the complete 719-line scanner pattern, with a byte cap."""
    def __init__(self, stream, guard):
        self._stream, self._guard = old._GuardedGzip(stream), guard
        self._lines = self._decoded = 0
        self._complete = False

    @property
    def decoded_bytes(self):
        return self._decoded

    @property
    def lines(self):
        return self._lines

    def readline(self, maximum=-1):
        self._guard()
        _require(not self._complete and type(maximum) is int and 0 < maximum <= MAX_LINE + 1,
                 'invalid bounded trace read')
        line = self._stream.readline(maximum)
        _require(type(line) is bytes and 0 < len(line) <= MAX_LINE and self._lines < 719,
                 'invalid trace line')
        self._lines += 1
        self._decoded += len(line)
        _require(self._decoded <= MAX_DECODED, 'decoded trace exceeds budget')
        return line

    def read(self, maximum=-1):
        self._guard()
        _require(not self._complete and maximum == 1 and self._lines == 719,
                 'trace must be scanned completely')
        end = self._stream.read(1)
        _require(end == b'', 'extra trace data')
        self._complete = True
        return end


class PrivateGame:
    def __init__(self, context, role, item, replay, coordinate, visit):
        self._context, self._item, self._replay = context, item, replay
        self._role, self._trace_attempted, self._trace_open = role, False, False
        self._coordinate, self._visit, self._semantic_complete = coordinate, visit, False

    def __repr__(self):
        return '<PrivateDaggerSourceGame>'

    def _open(self):
        _require(self._context is not None and self._context._active, 'source game is closed')

    @property
    def replay(self):
        self._open()
        return self._replay

    @property
    def identity(self):
        self._open()
        return dict(role=self._role, split=self._item.split, opponent=self._item.opponent, side=self._item.side)

    @property
    def game_limits(self):
        self._open()
        return dict(GAME_LIMITS)

    @property
    def task_sha256(self):
        self._open()
        return self._context._task_sha

    @property
    def crop_sha256(self):
        self._open()
        return self._context._crop_sha

    def make_agent(self):
        """Construct the authenticated baseline; caller owns its forward budget."""
        self._open()
        from learning.task_dagger_once_v1.policy import BoundedAgent
        return BoundedAgent(self._context._crop_path, self._context._crop_sha,
                            self._context._task_path, self._context._task_sha, limits=dict(GAME_LIMITS))

    def model_normalization(self):
        """Return independent copies of the authenticated task network metadata."""
        self._open()
        return self._context.model_normalization()

    @contextmanager
    def open_trace(self, *, guard):
        """Verify same-descriptor SHA before/after exactly one full scan."""
        self._open()
        _require(callable(guard) and not self._trace_attempted, 'trace scan already attempted')
        self._trace_attempted = True
        self._trace_open = True
        item = self._item
        path = old.fresh.ARTIFACTS / STAGE_NAME / 'traces' / (item.game + '.jsonl.gz')
        try:
            with old._open_checked(old.fresh.ARTIFACTS, path, old.COMPRESSED_MAX, guard) as raw:
                before, size = old._hash_stream(raw, old.COMPRESSED_MAX, guard)
                _require(before == item.hashes['trace'], 'trace digest differs')
                old._rewind(raw)
                with gzip.GzipFile(fileobj=raw, mode='rb') as stream:
                    reader = _TraceReader(stream, guard)
                    yield reader
                    _require(reader._complete and reader._lines == 719, 'trace scan incomplete')
                after, final_size = old._hash_stream(raw, old.COMPRESSED_MAX, guard)
                _require((after, final_size) == (before, size), 'trace changed during scan')
        finally:
            self._trace_open = False

    def scan_candidates(self, *, guard):
        """Authenticate, semantically scan and rank the complete source trace.

        Returns private ``(accounting, windows)``. A second visit is permitted
        only after this method succeeds and the game context exits normally.
        Merely consuming bytes through ``open_trace`` does not establish the
        full trace's model identity and controller-budget continuity.
        """
        self._open()
        from .sampling import RankedStream
        with self.open_trace(guard=guard) as trace:
            selector = RankedStream(trace, role=self._role, opponent=self._item.opponent,
                                    side=self._item.side, guard=guard)
            accounting = selector.scan(task_sha256=self.task_sha256, crop_sha256=self.crop_sha256)
            windows = selector.candidates()
        self._semantic_complete = True
        return accounting, windows

    def close(self):
        if self._context is not None:
            _require(not self._trace_open, 'trace scan is still active')
            self._replay, self._item = None, None
            self._context._active = False
            self._context = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, _type, _value, _traceback):
        if (_type is None and self._context is not None and self._semantic_complete
                and not self._trace_open):
            self._context._completed_scans[self._coordinate] = self._visit
        self.close()


def read_game(context, *, role, opponent, side, guard):
    """Read one already-completed coordinate, never a reserved/unknown game."""
    _require(isinstance(context, PrivateContext) and not context._active and callable(guard)
             and role in ROLES and opponent in OPPONENTS and type(side) is int and side in (0, 1),
             'source coordinate unavailable')
    coordinate = (role, opponent, side)
    item = context._entries[coordinate]
    visits = context._visits.get(coordinate, 0)
    _require(visits < 2 and (visits == 0 or context._completed_scans.get(coordinate) == 1),
             'source scan visit unavailable')
    context._visits[coordinate] = visits + 1
    stage = old.fresh.ARTIFACTS / STAGE_NAME
    context._active = True
    try:
        guard()
        # Everything is constructed exclusively from the 24-entry certificate.
        files = {
            'input': stage / 'job-inputs' / (item.game + '.json'),
            'row': stage / 'job-results' / (item.game + '.row.json'),
            'audit': stage / 'job-results' / (item.game + '.audit.json'),
            'completion': stage / 'job-results' / (item.game + '.complete.json'),
            'result': stage / 'games' / item.game / 'result.json',
            'wrapper': stage / 'agents' / (item.game + '.py'),
        }
        raw = {kind: _assert_blob(old.fresh.ARTIFACTS, path, item.hashes[kind], guard)
               for kind, path in files.items()}
        _verify_wrapper(raw['wrapper'], context._task_sha, context._crop_sha)
        _validate_receipts(item, _json(raw['input']), _json(raw['row']), _json(raw['audit']),
                           _json(raw['completion']), _json(raw['result']), context)
        trace = stage / 'traces' / (item.game + '.jsonl.gz')
        with old._open_checked(old.fresh.ARTIFACTS, trace, old.COMPRESSED_MAX, guard) as stream:
            digest, _size = old._hash_stream(stream, old.COMPRESSED_MAX, guard)
            _require(digest == item.hashes['trace'], 'trace digest differs')
        replay = old._read_replay(stage, item.game, item.hashes['replay'], item.seed, guard)
        guard()
        return PrivateGame(context, role, item, replay, coordinate, visits + 1)
    except old.SourceError:
        context._active = False
        raise SourceError('completed source file verification failed') from None
    except BaseException:
        context._active = False
        raise
