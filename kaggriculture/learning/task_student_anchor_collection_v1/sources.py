"""Authenticate only already-completed Anchor baseline games.

The caller owns process, memory and wall limits and must bind this module in a
new registration. No old Anchor registration, seed list, inventory, worker,
model inference or official game is invoked here. The original ledger, seeds,
game names, replay, model paths and checkpoints are private in-memory values.

Two fixed batches reuse the existing six-role DAgger sampler. The source-role
alias A01..A10 is stable across batches and must be retained in any combined
data plan: the sampler's T01/V01 aliases alone are not globally unique.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import json
import subprocess
from types import MappingProxyType

from learning.task_student_dagger_collection_v1 import sources as dagger


old = dagger.old
SourceError = dagger.SourceError
ANCHOR_COMMIT = '1610ee975de25c0c1afa7a34facd54f1904ee331'
ANCHOR_ROOT_SHA256 = '6a11ff4cf5bac0301ffb8b2cece61ea290fba71ce36b09a748a99a98a15356fa'
ANCHOR_PREFIX = 'review/evidence/task-anchor-reentry-v1-20261027/'
FRESH_RESULT_COMMIT = '0e9734a0109a0ee4822ef1310e8b56fac21eaa48'
FRESH_RESULT = 'review/evidence/task-pair-fresh-multi-v1-result-20261031/ledger.json'
STAGE_NAME = 'task-anchor-reentry-v1-20261027'
TRAIN_ROLES, DEV_ROLES = dagger.TRAIN_ROLES, dagger.DEV_ROLES
ROLES, OPPONENTS = dagger.ROLES, dagger.OPPONENTS
SOURCE_ROLES = tuple('A%02d' % number for number in range(1, 11))
SLOTS = {
    1: TRAIN_ROLES + DEV_ROLES,
    2: TRAIN_ROLES[:2] + DEV_ROLES,
}
RECEIPTS = frozenset(('row', 'audit', 'complete', 'storage'))
_require = dagger._require
_digest = dagger._digest
_sha = dagger._sha


def _json(raw, expected=dict, maximum=dagger.MAX_SMALL):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, 'invalid source document size')
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, 'duplicate source document key')
            value[key] = item
        return value
    def nonfinite(_value):
        raise SourceError('nonfinite source document value')
    try:
        value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError):
        raise SourceError('invalid source document') from None
    _require(type(value) is expected, 'unexpected source document type')
    return value


def _git_blob(commit, relative, guard):
    guard()
    _require((commit == ANCHOR_COMMIT and relative in {
        ANCHOR_PREFIX + 'root-result-acceptance.json',
        ANCHOR_PREFIX + 'independent/final-verification.json',
        ANCHOR_PREFIX + 'freeze.json', ANCHOR_PREFIX + 'ledger.json',
    }) or (commit == FRESH_RESULT_COMMIT and relative == FRESH_RESULT),
        'source blob not allowed')
    try:
        result = subprocess.run(['git', '-C', str(old.fresh.REPO), 'show', commit + ':' + relative],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=30, check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        raise SourceError('source blob unavailable') from None
    guard()
    _require(result.returncode == 0 and 0 < len(result.stdout) <= dagger.MAX_SMALL,
             'source blob unavailable')
    return result.stdout


@dataclass(frozen=True, repr=False)
class _Entry:
    game: str
    seed: int
    split: str                 # Historical Anchor dev/holdout, not new split.
    opponent: str
    side: int
    source_role: str
    ledger_row_sha256: str
    hashes: object

    def __repr__(self):
        return '<PrivateAnchorCompletedEntry>'


class PrivateContext(dagger.PrivateContext):
    def __repr__(self):
        return '<PrivateAnchorSourceContext>'

    @property
    def task_sha256(self):
        """Private model identity, for a local binding only."""
        return self._task_sha

    @property
    def crop_sha256(self):
        """Private model identity, for a local binding only."""
        return self._crop_sha

    def private_role_binding(self):
        """Copy source-role/seed bindings for local salted commitments only."""
        mapping = {}
        for item in self._entries.values():
            previous = mapping.setdefault(item.source_role, item.seed)
            _require(previous == item.seed, 'Anchor role mapping differs')
        _require(set(mapping) == set(SOURCE_ROLES) and len(set(mapping.values())) == 10,
                 'Anchor role mapping incomplete')
        return {role: mapping[role] for role in SOURCE_ROLES}

    def source_role(self, *, batch, role):
        """Return the A alias, or None for the two explicit empty slots."""
        _require(type(batch) is int and batch in SLOTS and role in ROLES, 'source batch slot unavailable')
        if role not in SLOTS[batch]:
            return None
        return self._entries[(batch, role, OPPONENTS[0], 0)].source_role


def _fresh_completed_seeds(guard):
    """Read only the already-completed four-role binding, never its inventory."""
    reg = dagger._json(dagger._git_blob(dagger.FRESH_COMMIT, dagger.FRESH_REGISTRATION, guard))
    public = _json(_git_blob(FRESH_RESULT_COMMIT, FRESH_RESULT, guard))
    _require(public.get('schema') == 'task-pair-fresh-multi-public-v1'
             and public.get('registration_commit') == dagger.FRESH_COMMIT
             and type(public.get('games')) is int and public['games'] == 32
             and type(public.get('official_replays')) is int and public['official_replays'] == 32
             and type(public.get('seed_clusters')) is int and public['seed_clusters'] == 4
             and public.get('training') is False and public.get('promotion') is False
             and type(public.get('by_seed')) is dict
             and set(public['by_seed']) == {'F01', 'F02', 'F03', 'F04'},
             'completed fresh public result differs')
    binding_raw = old._small(old.fresh.PRIVATE_ROOT, old.fresh.PRIVATE / 'binding.local.json', 65536, guard)
    _require(_sha(binding_raw) == _digest(reg.get('binding_commitment_sha256')),
             'completed fresh binding differs')
    binding = _json(binding_raw, maximum=65536)
    _require(binding.get('schema') == 'task-pair-fresh-multi-private-binding-v1'
             and binding.get('cohorts') == old.entry.COHORTS and type(binding.get('seed_roles')) is dict
             and set(binding['seed_roles']) == {'F01', 'F02', 'F03', 'F04'},
             'completed fresh roles differ')
    values = tuple(binding['seed_roles'].values())
    _require(all(type(seed) is int and 0 <= seed < 2**32 for seed in values)
             and len(set(values)) == 4, 'invalid completed fresh roles')
    market = old._small(old.fresh.PRIVATE_ROOT,
        old.fresh.PRIVATE_ROOT / 'task-pair-market-consumed-v1-20261031' / 'binding.local.json', 65536, guard)
    _require(_sha(market) == _digest(binding.get('market_binding_sha256')),
             'completed fresh market binding differs')
    campaign = _json(old._small(old.fresh.PRIVATE_ROOT, old.fresh.PRIVATE / 'campaign-result.local.json',
                               old.RESULT_MAX, guard))
    closed = _json(old._small(old.fresh.PRIVATE_ROOT, old.fresh.PRIVATE / 'closed.local.json',
                             old.SMALL_MAX, guard))
    # This is an in-memory verifier. Do not call fresh.verify_binding or
    # fresh.load_verified_market_binding: they parse historical inventories.
    try:
        old._verify_campaign(reg, binding_raw, market, campaign, closed)
    except old.SourceError:
        raise SourceError('completed fresh campaign differs') from None
    return frozenset(values)


def _review_entries(ledger, stats, context, dagger_seeds, fresh_seeds):
    _require(type(ledger) is list and len(ledger) == 120 and type(stats) is dict,
             'completed Anchor population differs')
    evidence = stats.get('all_job_evidence_sha256')
    _require(type(evidence) is dict and len(evidence) == 120, 'Anchor job evidence incomplete')
    baseline = [row for row in ledger if type(row) is dict and row.get('policy') == 'baseline']
    _require(len(baseline) == 40, 'completed baseline population differs')
    first, splits, indexed, names = [], {}, {}, set()
    for row in baseline:
        seed, split, game = row.get('seed'), row.get('split'), row.get('game')
        opponent, side = row.get('opponent'), row.get('side')
        _require(type(seed) is int and 0 <= seed < 2**32 and split in ('dev', 'holdout')
                 and type(game) is str and dagger.GAME_NAME.fullmatch(game) and game not in names
                 and opponent in OPPONENTS and type(side) is int and side in (0, 1)
                 and row.get('passed') is True and type(row.get('states')) is int and row['states'] == 720
                 and type(row.get('action_turns')) is int and row['action_turns'] == 719
                 and row.get('status') == ['DONE', 'DONE'] and type(row.get('agent_errors')) is int
                 and row['agent_errors'] == 0 and row.get('engine_sha256') == dagger.ENGINE_SHA
                 and row.get('task_checkpoint_sha256') == context._task_sha
                 and row.get('crop_checkpoint_sha256') == context._crop_sha,
                 'completed baseline identity differs')
        if seed not in splits:
            first.append(seed)
            splits[seed] = split
        _require(splits[seed] == split, 'historical seed appears in both splits')
        coordinate = (seed, opponent, side)
        _require(coordinate not in indexed, 'duplicate completed Anchor coordinate')
        hashes = evidence.get(game)
        _require(type(hashes) is dict and set(hashes) == RECEIPTS, 'Anchor receipt evidence differs')
        hashes = {key: _digest(hashes[key]) for key in RECEIPTS}
        hashes['completion'] = hashes.pop('complete')
        for key, field in (('replay', 'replay_gzip_sha256'), ('trace', 'trace_sha256'),
                           ('wrapper', 'bot_sha256')):
            hashes[key] = _digest(row.get(field))
        _digest(row.get('rival_sha256'))
        ledger_row = {key: value for key, value in row.items() if key != 'first_request_difference_step'}
        ledger_sha = _sha(json.dumps(ledger_row, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':'), allow_nan=False).encode('ascii'))
        indexed[coordinate] = (game, seed, split, opponent, side, ledger_sha, MappingProxyType(hashes))
        names.add(game)
    _require(len(first) == 10 and not set(first) & set(dagger_seeds)
             and not set(first) & set(fresh_seeds), 'historical completed roles overlap')
    _require(set(indexed) == {(seed, opponent, side) for seed in first
                              for opponent in OPPONENTS for side in (0, 1)},
             'completed Anchor panel is incomplete')
    original_dev = [seed for seed in first if splits[seed] == 'dev']
    original_holdout = [seed for seed in first if splits[seed] == 'holdout']
    _require(len(original_dev) == 2 and len(original_holdout) == 8,
             'historical completed split counts differ')
    new_train, new_dev = original_holdout[:6], original_dev + original_holdout[6:]
    # Public A labels follow the fixed new split, not historical ledger order:
    # A01–A06 training, A07–A10 development; order inside each historical
    # source split is the first appearance in the completed baseline ledger.
    source_alias = dict(zip(new_train + new_dev, SOURCE_ROLES))
    assignment = {
        1: dict(zip(TRAIN_ROLES + DEV_ROLES, new_train[:4] + new_dev[:2])),
        2: dict(zip(TRAIN_ROLES[:2] + DEV_ROLES, new_train[4:] + new_dev[2:])),
    }
    entries = {}
    for batch, slots in assignment.items():
        _require(tuple(slots) == SLOTS[batch], 'Anchor batch assignment differs')
        for role, seed in slots.items():
            for opponent in OPPONENTS:
                for side in (0, 1):
                    game, seed, split, opponent, side, ledger_sha, hashes = indexed[(seed, opponent, side)]
                    entries[(batch, role, opponent, side)] = _Entry(
                        game, seed, split, opponent, side, source_alias[seed], ledger_sha, hashes)
    _require(len(entries) == 40, 'Anchor batch population differs')
    return entries


def open_context(*, guard):
    """Authenticate public roots, historical evidence and the frozen model."""
    _require(callable(guard), 'resource guard required')
    # This existing verifier reads only completed DAgger entries and the
    # already-public fresh model commitment, never an unused-seed manifest.
    base = dagger.open_context(guard=guard)
    dagger_seeds = {entry.seed for entry in base._entries.values()}
    _require(len(dagger_seeds) == 6, 'completed DAgger roles differ')
    fresh_seeds = _fresh_completed_seeds(guard)
    _require(not dagger_seeds & set(fresh_seeds), 'completed historical roles overlap')
    root_raw = _git_blob(ANCHOR_COMMIT, ANCHOR_PREFIX + 'root-result-acceptance.json', guard)
    _require(_sha(root_raw) == ANCHOR_ROOT_SHA256, 'published Anchor root differs')
    root = _json(root_raw)
    _require(root.get('passed') is True and root.get('engineering_complete') is True
             and type(root.get('completed_games')) is int and root['completed_games'] == 120,
             'Anchor completion root differs')
    verify_raw = _git_blob(ANCHOR_COMMIT, ANCHOR_PREFIX + 'independent/final-verification.json', guard)
    _require(_sha(verify_raw) == _digest(root.get('independent_verification_sha256')),
             'Anchor independent verification digest differs')
    verify = _json(verify_raw)
    _require(verify.get('passed') is True, 'Anchor independent verification failed')
    stage = old.fresh.ARTIFACTS / STAGE_NAME
    report_raw = old._small(old.fresh.ARTIFACTS, stage / 'independent/final-result-review.json',
                            dagger.MAX_SMALL, guard)
    stats_raw = old._small(old.fresh.ARTIFACTS, stage / 'independent/final-statistics.json',
                           dagger.MAX_SMALL, guard)
    _require(_sha(report_raw) == _digest(verify.get('report_sha256'))
             and _sha(report_raw) == _digest(root.get('independent_result_sha256'))
             and _sha(stats_raw) == _digest(verify.get('statistics_sha256')),
             'Anchor private review chain differs')
    report, stats = _json(report_raw), _json(stats_raw)
    _require(report.get('review_passed') is True and report.get('all_registered_jobs_completed') is True
             and type(report.get('completed_games')) is int and report['completed_games'] == 120
             and report.get('final_statistics_sha256') == _sha(stats_raw)
             and stats.get('passed') is True, 'Anchor private review failed')
    freeze_raw = _git_blob(ANCHOR_COMMIT, ANCHOR_PREFIX + 'freeze.json', guard)
    _require(_sha(freeze_raw) == _digest(root.get('freeze_sha256'))
             and _sha(freeze_raw) == _digest(report.get('freeze_sha256')),
             'Anchor freeze digest differs')
    freeze = _json(freeze_raw)
    local, public = freeze.get('runtime_files'), freeze.get('published_runtime_files')
    _require(type(local) is dict and type(public) is dict, 'Anchor runtime sources missing')
    for relative in dagger.RUNTIME_COMPONENTS:
        dagger._assert_blob(old.fresh.GAME, old.fresh.GAME / relative, local.get(relative), guard)
        dagger._assert_blob(old.fresh.REPO, old.fresh.REPO / 'kaggriculture' / relative,
                            public.get(relative), guard)
    ledger = _json(_git_blob(ANCHOR_COMMIT, ANCHOR_PREFIX + 'ledger.json', guard), expected=list)
    entries = _review_entries(ledger, stats, base, dagger_seeds, fresh_seeds)
    return PrivateContext(entries, base._task_path, base._task_sha, base._crop_path, base._crop_sha)


class PrivateGame(dagger.PrivateGame):
    def __repr__(self):
        return '<PrivateAnchorSourceGame>'

    @property
    def identity(self):
        self._open()
        return dict(source='anchor', source_role=self._item.source_role, batch=self._coordinate[0],
                    role=self._role, split='train' if self._role in TRAIN_ROLES else 'dev',
                    opponent=self._item.opponent, side=self._item.side)

    @contextmanager
    def open_trace(self, *, guard):
        """Same-descriptor trace verification for an authenticated Anchor game."""
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
                    reader = dagger._TraceReader(stream, guard)
                    yield reader
                    _require(reader._complete and reader._lines == 719, 'trace scan incomplete')
                after, final_size = old._hash_stream(raw, old.COMPRESSED_MAX, guard)
                _require((after, final_size) == (before, size), 'trace changed during scan')
        finally:
            self._trace_open = False


def read_game(context, *, batch, role, opponent, side, guard):
    """Read a unique completed coordinate; each is attempted at most twice."""
    _require(isinstance(context, PrivateContext) and not context._active and callable(guard)
             and type(batch) is int and batch in SLOTS and role in SLOTS[batch]
             and opponent in OPPONENTS and type(side) is int and side in (0, 1),
             'source coordinate unavailable')
    coordinate = (batch, role, opponent, side)
    item = context._entries[coordinate]
    visits = context._visits.get(coordinate, 0)
    _require(visits < 2 and (visits == 0 or context._completed_scans.get(coordinate) == 1),
             'source scan visit unavailable')
    context._visits[coordinate] = visits + 1
    context._active = True
    stage = old.fresh.ARTIFACTS / STAGE_NAME
    try:
        guard()
        raw = {}
        for kind in ('row', 'audit', 'completion', 'storage'):
            suffix = 'complete' if kind == 'completion' else kind
            raw[kind] = dagger._assert_blob(old.fresh.ARTIFACTS,
                stage / 'job-results' / (item.game + '.' + suffix + '.json'), item.hashes[kind], guard)
        wrapper = dagger._assert_blob(old.fresh.ARTIFACTS,
            stage / 'agents' / (item.game + '.py'), item.hashes['wrapper'], guard)
        dagger._verify_wrapper(wrapper, context._task_sha, context._crop_sha)
        row, audit, complete, storage = (_json(raw[kind]) for kind in ('row', 'audit', 'completion', 'storage'))
        row_sha = _sha(json.dumps(row, sort_keys=True, ensure_ascii=True,
                                 separators=(',', ':'), allow_nan=False).encode('ascii'))
        _require(row_sha == item.ledger_row_sha256, 'Anchor completed ledger row differs')
        output = storage.get('output')
        _require(storage.get('passed') is True and storage.get('error_kind') is None
                 and type(output) is dict and type(output.get('refused')) is int and output['refused'] == 0
                 and type(output.get('used_bytes')) is int and type(output.get('limit_bytes')) is int
                 and 0 <= output['used_bytes'] <= output['limit_bytes'] and output['limit_bytes'] > 0,
                 'Anchor storage receipt differs')
        # These two producer files are not individually SHA-bound by the
        # historical independent certificate; they are semantic cross-checks.
        job = _json(old._small(old.fresh.ARTIFACTS, stage / 'job-inputs' / (item.game + '.json'),
                               dagger.MAX_SMALL, guard))
        result = _json(old._small(old.fresh.ARTIFACTS, stage / 'games' / item.game / 'result.json',
                                  dagger.MAX_SMALL, guard))
        dagger._validate_receipts(item, job, row, audit, complete, result, context)
        dagger._assert_blob(old.fresh.ARTIFACTS, stage / 'sources' / (item.opponent + '.py'),
                            row.get('rival_sha256'), guard)
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
