"""Fail-closed exclusion of prior panel seeds for a future, separate panel.

The returned inventory contains private seed values in memory. Do not log or
publish it; its repr exposes only a count. The committed evidence anchors and
the private document remain distinct from the future panel's own directory.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

from learning.task_teacher_merged_eval_v1 import experiment as old

PRIOR_STAGE = 'task-teacher-merged-eval-v1-20261028'
PRIOR_COMMIT = '692b9c7cd640c31bfb319e7ae434c6fa7176ba43'
FAILURE_COMMIT = '7789fb378d2f7779b23909f0cfd8d11beb764fc0'
EVIDENCE = 'review/evidence/' + PRIOR_STAGE + '/'
MIB = 1024 ** 2
MAX_TOTAL = 256 * MIB
MAX_FILE = 16 * MIB
NAMES = frozenset(('registration.json', 'input-manifest.json', 'root-decision.json',
    'seeds.json', 'seed-inventory.json', 'seed-inventory.local.json', 'seed-roles.local.json',
    'seeds.local.json', 'private-seeds.local.json'))
PRIVATE_NAMES = frozenset(('seeds.json', 'seeds.local.json', 'private-seeds.local.json',
                           'seed-roles.local.json'))
NON_METADATA_DIRS = frozenset(('games', 'traces', 'job-results', 'job-acceptance',
    'job-supervision', 'agents', 'replay', 'replays', 'results'))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate metadata field')
        result[key] = value
    return result


def _json(data, limit=MAX_FILE):
    if not isinstance(data, bytes) or len(data) > limit:
        raise ValueError('Invalid metadata size')
    def invalid_constant(_):
        raise ValueError('Invalid numeric metadata')
    try:
        return json.loads(data.decode('utf-8-sig'), object_pairs_hook=_unique, parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid metadata') from exc


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _verify_consumed(*, private_document, prior_inventory, local_registration, local_freeze,
                     local_failure, local_execution, committed_registration,
                     committed_freeze, committed_certificate):
    """Verify snapshots against pinned commit blobs; return values only in memory.

    The public caller obtains committed blobs from fixed Git commits. Keeping
    verification separate lets regression tests use entirely synthetic data.
    """
    snapshots = (private_document, prior_inventory, local_registration, local_freeze,
                 local_failure, local_execution, committed_registration, committed_freeze, committed_certificate)
    if any(not isinstance(value, bytes) or len(value) > MAX_FILE for value in snapshots):
        raise ValueError('Historical evidence missing or oversized')
    if len(private_document) > 16384:
        raise ValueError('Historical seed document oversized')
    if local_registration != committed_registration or local_freeze != committed_freeze:
        raise ValueError('Historical registration differs from committed evidence')
    cfg = _json(committed_registration)
    freeze = _json(committed_freeze)
    cert = _json(committed_certificate)
    failure = _json(local_failure)
    execution = _json(local_execution)
    if not all(isinstance(v, dict) for v in (cfg, freeze, cert, failure, execution)):
        raise ValueError('Invalid historical evidence')
    registration_sha = _sha(committed_registration)
    freeze_sha = _sha(committed_freeze)
    if (freeze.get('registration_sha256') != registration_sha
        or execution.get('commit') != PRIOR_COMMIT
        or execution.get('registration_sha256') != registration_sha
        or execution.get('freeze_sha256') != freeze_sha
        or cert.get('execution_commit') != PRIOR_COMMIT
        or cert.get('registration_sha256') != registration_sha
        or cert.get('freeze_sha256') != freeze_sha
        or cert.get('local_failure_sha256') != _sha(local_failure)
        or cert.get('status') != 'incomplete_preflight_stop'
        or cert.get('stop_reason') != 'available_memory_preflight'
        or cert.get('all_four_seed_roles_consumed') is not True
        or cert.get('retry_allowed') is not False
        or type(cert.get('planned_games')) is not int or cert['planned_games'] != 72
        or type(cert.get('attempted_games')) is not int or cert['attempted_games'] != 26
        or type(cert.get('accepted_games')) is not int or cert['accepted_games'] != 26
        or type(cert.get('unstarted_games')) is not int or cert['unstarted_games'] != 46
        or cert.get('next_game_input_present') is not False
        or cert.get('economic_evaluation_started') is not False
        or failure.get('stop_reason') != cert['stop_reason']
        or failure.get('retry_allowed') is not False
        or type(failure.get('attempted_games')) is not int or failure['attempted_games'] != 26
        or type(failure.get('completed_games')) is not int or failure['completed_games'] != 26
        or failure.get('uncertain_current_game') is not None):
        raise ValueError('Historical failure evidence is not bound')
    if (type(cfg.get('seed_count')) is not int or cfg['seed_count'] != 4
        or cfg.get('seed_document_sha256') != _sha(private_document)
        or cfg.get('seed_inventory_sha256') != _sha(prior_inventory)
        or cfg.get('seed_commitment_domain') != old.SEED_DOMAIN.decode('ascii').rstrip('\x00')):
        raise ValueError('Historical seed document identity changed')
    document = _json(private_document, 16384)
    inventory = _json(prior_inventory)
    seeds, _ = old.validate_seed_document(document)
    if (old.seed_commitment(document) != cfg.get('seed_commitment')
        or not isinstance(inventory, dict) or not isinstance(inventory.get('excluded_uint32'), list)
        or any(type(value) is not int or not 0 <= value < 2**32 for value in inventory['excluded_uint32'])
        or set(seeds) & set(inventory['excluded_uint32'])):
        raise ValueError('Historical seed commitment or inventory changed')
    return frozenset(seeds)


def _read_regular(path, boundary, limit=MAX_FILE):
    """Reject existing links before opening an allowed metadata file."""
    path, boundary = Path(path), Path(boundary)
    if not path.is_relative_to(boundary):
        raise ValueError('Metadata path outside boundary')
    for part in (path, *path.parents):
        old._reject_reparse(part)
        if part == boundary:
            break
    if not path.is_file() or not path.resolve().is_relative_to(boundary.resolve()):
        raise ValueError('Metadata file missing or outside boundary')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Metadata file exceeds budget')
    return data


def _committed(name, commit):
    return old.git('show', commit + ':' + EVIDENCE + name)


def _load_consumed_values():
    """Load the four consumed prior seeds only after pinned Git/local checks.

    Git ancestry is required; the local failure receipt and private document
    are never returned or written. Exceptions contain no private values.
    """
    old.git('merge-base', '--is-ancestor', PRIOR_COMMIT, FAILURE_COMMIT)
    old.git('merge-base', '--is-ancestor', FAILURE_COMMIT, 'HEAD')
    out = old.trusted_out(old.OUT)
    if out.name != PRIOR_STAGE:
        raise ValueError('Unexpected historical stage')
    return _verify_consumed(
        private_document=_read_regular(out/'private-seeds.local.json', out, 16384),
        prior_inventory=_read_regular(out/'seed-inventory.local.json', out),
        local_registration=_read_regular(out/'registration.json', out),
        local_freeze=_read_regular(out/'freeze.json', out),
        local_failure=_read_regular(out/'games-failure.local.json', out),
        local_execution=_read_regular(out/'pre-execution-commit.json', out),
        committed_registration=_committed('registration.json', PRIOR_COMMIT),
        committed_freeze=_committed('freeze.json', PRIOR_COMMIT),
        committed_certificate=_committed('failure-certificate.json', FAILURE_COMMIT))


def _stage_path(artifacts, new_out):
    if '..' in Path(artifacts).parts or '..' in Path(new_out).parts:
        raise ValueError('Artifact traversal refused')
    artifacts, new_out = Path(os.path.abspath(artifacts)), Path(os.path.abspath(new_out))
    if new_out.parent != artifacts or not new_out.name or new_out.name.casefold() == PRIOR_STAGE.casefold():
        raise ValueError('Future stage must be a distinct direct artifact child')
    for part in (artifacts, *artifacts.parents):
        old._reject_reparse(part)
    old._reject_reparse(new_out)
    if not artifacts.is_dir() or artifacts.resolve() != artifacts:
        raise ValueError('Untrusted artifact root')
    if new_out.exists() and (not new_out.is_dir() or new_out.resolve() != new_out):
        raise ValueError('Untrusted future stage')
    return artifacts, new_out


def _scan_metadata(artifacts, public, new_out, tracked):
    """Scan recognized metadata and job plans, never games, results or replay."""
    artifacts, new_out = _stage_path(artifacts, new_out)
    public = Path(os.path.abspath(public))
    for part in (public, *public.parents):
        old._reject_reparse(part)
    if not public.is_dir() or public.resolve() != public:
        raise ValueError('Untrusted public root')
    files = []
    for name in tracked:
        if not isinstance(name, str):
            raise ValueError('Invalid tracked metadata path')
        path = PurePosixPath(name)
        if ('\\' in name or ':' in name or path.is_absolute() or path.as_posix() != name
            or not path.parts or any(part in ('.', '..') for part in path.parts)):
            raise ValueError('Invalid tracked metadata path')
        if (len(path.parts) >= 3 and tuple(p.casefold() for p in path.parts[:2]) == ('review', 'evidence')
            and path.parts[2].casefold() == new_out.name.casefold()):
            raise ValueError('Future stage name already has tracked public evidence')
        if {p.casefold() for p in path.parts} & (NON_METADATA_DIRS | {'private'}):
            continue
        if (path.name.lower() in NAMES
            or (path.parent.name.casefold() == 'job-inputs' and path.name.lower().endswith('.json'))):
            files.append((public.joinpath(*path.parts), public))
    def walk_error(error):
        raise ValueError('Historical metadata cannot be scanned') from error
    for folder, dirs, names in os.walk(artifacts, followlinks=False, onerror=walk_error):
        directory = Path(folder)
        kept = []
        for name in dirs:
            child = directory/name
            old._reject_reparse(child)
            if child != new_out and name.casefold() not in NON_METADATA_DIRS:
                kept.append(name)
        dirs[:] = kept
        private = 'private' in {part.casefold() for part in directory.relative_to(artifacts).parts}
        for name in names:
            lower = name.lower()
            if not lower.endswith('.json') or (private and lower not in PRIVATE_NAMES):
                continue
            if lower in NAMES or (not private and directory.name.casefold() == 'job-inputs'):
                files.append((directory/name, artifacts))
    seen, total = set(), 0
    for path, boundary in files:
        data = _read_regular(path, boundary)
        total += len(data)
        if total > MAX_TOTAL:
            raise ValueError('Historical metadata exceeds budget')
        seen.update(old._seed_values(_json(data), path.name.lower() == 'seeds.json'))
    return seen, len(files), total


class SeedInventory:
    """Private seed exclusion set; its representation never contains values."""
    def __init__(self, values, metadata_files, metadata_bytes, parent_commit=None):
        values = tuple(values)
        if (any(type(value) is not int or not 0 <= value < 2**32 for value in values)
            or type(metadata_files) is not int or metadata_files < 0
            or type(metadata_bytes) is not int or metadata_bytes < 0
            or (parent_commit is not None and (not isinstance(parent_commit, str)
                or len(parent_commit) != 40 or any(c not in '0123456789abcdef' for c in parent_commit)))):
            raise ValueError('Invalid inventory')
        self._values = frozenset(values)
        self.metadata_files, self.metadata_bytes, self.parent_commit = metadata_files, metadata_bytes, parent_commit

    @property
    def count(self):
        return len(self._values)

    def __repr__(self):
        return 'SeedInventory(count=' + str(self.count) + ')'

    def contains(self, value):
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError('Invalid candidate seed')
        return value in self._values

    def require_unused(self, values):
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError('Invalid candidate seeds')
        if (any(type(value) is not int or not 0 <= value < 2**32 for value in values)
            or len(set(values)) != len(values) or set(values) & self._values):
            raise ValueError('Candidate seeds collide with history')

    def private_values(self):
        """Return values for an explicitly private local manifest, never logs."""
        return tuple(sorted(self._values))


def load_consumed_prior():
    """Return a redacted inventory after verifying the consumed prior panel."""
    return SeedInventory(_load_consumed_values(), 0, 0)


def build_inventory(new_out):
    """Build once before creating or publishing the future stage.

    The caller must exclusively create the stage afterwards and freeze this
    private inventory.  Once stage evidence is committed, its execution path
    must validate the frozen inventory rather than calling this function again.
    """
    consumed = load_consumed_prior()
    artifacts, new_out = _stage_path(old.trusted_artifacts(), new_out)
    if old.git('status', '--porcelain').strip():
        raise ValueError('Public repository must be clean before inventory')
    parent_commit = old.git('rev-parse', 'HEAD').decode('ascii').strip()
    if len(parent_commit) != 40 or any(c not in '0123456789abcdef' for c in parent_commit):
        raise ValueError('Invalid inventory parent commit')
    try:
        old.git('merge-base', '--is-ancestor', FAILURE_COMMIT, parent_commit)
    except subprocess.CalledProcessError:
        raise ValueError('Inventory parent does not contain the consumed panel') from None
    public_stage = old.PUBLIC/'review/evidence'/new_out.name
    for path in (public_stage, public_stage.parent, public_stage.parent.parent, old.PUBLIC):
        old._reject_reparse(path)
    if new_out.exists() or public_stage.exists():
        raise ValueError('Future stage name is already occupied')
    tracked = [name for name in old.git('ls-files', '-z').decode('utf-8').split('\0') if name]
    historical, count, size = _scan_metadata(artifacts, old.PUBLIC, new_out, tracked)
    _stage_path(artifacts, new_out)
    for path in (public_stage, public_stage.parent, public_stage.parent.parent, old.PUBLIC):
        old._reject_reparse(path)
    if (old.git('rev-parse', 'HEAD').decode('ascii').strip() != parent_commit
        or old.git('status', '--porcelain').strip() or new_out.exists() or public_stage.exists()):
        raise ValueError('Repository or stage changed during inventory')
    return SeedInventory(historical | consumed._values, count, size, parent_commit)
