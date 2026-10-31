"""One bounded comparison of already-consumed teacher training groups.

The archive reader never decodes seed members. This is a data-overlap
diagnostic, not model inference, training, a new game, or an independent test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time

from learning.task_student_continuation_v1 import experiment as previous
from learning.task_multi_teacher_probe_v1 import experiment as supervisor
from learning.task_teacher_transfer_dataset_v1 import experiment as transfer


REPO = previous.REPO
GAME = REPO.parent.parent / 'kaggriculture'
PRIVATE = REPO.parent / 'private' / 'task-teacher-cross-source-v1-20261031'
REGISTRATION = REPO / 'review/evidence/task-teacher-cross-source-v1-20261031/registration.json'
BASE_COMMIT = 'f6da6efe095e48c933558ea6243650fa6c063a10'
HIST_COMMIT = 'd8a1c2ff025991f724ee1ca5220aac4b27f277a4'
HIST_OLD_REL = 'review/evidence/task-teacher-choice-bc-v1-20261027/train-data-result.json'
HIST_NEW_REL = 'review/evidence/task-teacher-refresh-v2-20261028/data-seal.json'
DATA_COMMIT = '7a2389ea77822a9bf4f11d3f9dc244765d270041'
OLD_PATHS = {
    'old': GAME / 'artifacts/task-teacher-choice-bc-v1-20261027/private/train.npz',
    'new': GAME / 'artifacts/task-teacher-refresh-v2-20261028/private/new-teacher.npz',
}
SCHEMA_PATHS = (
    GAME / 'artifacts/task-teacher-choice-bc-v1-20261027/schema.json',
    GAME / 'artifacts/task-teacher-refresh-v2-20261028/schema.json',
)
GROUPS = {'old': 2253, 'new': 2851}
CANDIDATES = {'old': 118955, 'new': 203196}
JOB = dict(supervisor.JOB)
MEMORY_INITIAL, DISK_FLOOR = 5 * 1024**3, 4 * 1024**3
MAX_METADATA = 1024**2
SOURCES = tuple('kaggriculture/learning/' + name for name in (
    'task_teacher_cross_source_v1/__init__.py', 'task_teacher_cross_source_v1/archive.py',
    'task_teacher_cross_source_v1/overlap.py', 'task_teacher_cross_source_v1/experiment.py',
    'task_teacher_transfer_dataset_v1/experiment.py', 'task_teacher_transfer_dataset_v1/dataset.py',
    'task_student_anchor_collection_v1/sources.py', 'task_student_dagger_collection_v1/sources.py',
    'task_student_continuation_v1/experiment.py', 'task_multi_teacher_probe_v1/experiment.py',
    'task_teacher_dev_deviation_v1/membership.py', 'task_pair_fresh_multi_v1/entry.py',
    'task_bc_v1/model.py', 'task_choice_v1/features.py')) + ('kaggriculture/decision/api.py',)


def _git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args],
                                   stderr=subprocess.DEVNULL, timeout=30).decode('utf-8').strip()


def _windows():
    return os.name == 'nt'


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit=MAX_METADATA):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Invalid bounded source')
    with path.open('rb') as source:
        raw = source.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ValueError('Bounded source size differs')
    return raw


def _hash_field(raw, key):
    """Extract only one named digest from an already-consumed source receipt."""
    if type(raw) is not bytes or len(raw) > MAX_METADATA or not key.isidentifier():
        raise ValueError('Invalid historical metadata request')
    matches = re.findall(rb'"' + key.encode('ascii') + rb'"\s*:\s*"([a-f0-9]{64})"', raw)
    if len(matches) != 1:
        raise ValueError('Historical metadata field differs')
    return matches[0].decode('ascii')


def _historical_metadata():
    """Extract only digests from the two completed training-source receipts.

    The old receipt can include counts keyed by already-consumed train seeds;
    those fields are neither parsed nor printed. No holdout inventory or
    historical registration containing possible reserved values is opened.
    """
    blobs = [subprocess.check_output(['git', '-C', str(REPO), 'show', HIST_COMMIT + ':' + relative],
                                    stderr=subprocess.DEVNULL, timeout=30)
             for relative in (HIST_OLD_REL, HIST_NEW_REL)]
    if any(not raw or len(raw) > MAX_METADATA for raw in blobs):
        raise ValueError('Historical source receipt differs')
    return dict(old_receipt_sha256=_sha(blobs[0]), new_receipt_sha256=_sha(blobs[1]),
                archive_sha256={'old': _hash_field(blobs[0], 'array_sha256'),
                                'new': _hash_field(blobs[1], 'array_sha256')},
                schema_sha256=_hash_field(blobs[1], 'schema_sha256'))


def _schema_identity(expected_raw_sha, metadata):
    """Bind the two old ordered schemas to the authenticated baseline schema."""
    from learning.task_bc_v1.model import DIMENSIONS
    from learning.task_choice_v1.features import OPS, schema_hash
    old, new = (_read(path) for path in SCHEMA_PATHS)
    if old != new or _sha(old) != expected_raw_sha:
        raise ValueError('Historical schema bytes differ')
    schema = previous._document(old)
    keys = ('schema_version', 'schema_sha256', 'shared_feature_names', 'feature_names', 'ops')
    if (set(schema) != set(keys) or type(metadata) is not dict
            or any(schema.get(key) != metadata.get(key) for key in keys)
            or schema['ops'] != list(OPS) or metadata.get('dimensions') != list(DIMENSIONS)
            or len(schema['shared_feature_names']) != 275 or len(schema['feature_names']) != 98
            or schema['schema_sha256'] != schema_hash(schema['feature_names'], schema['shared_feature_names'])):
        raise ValueError('Historical and current semantic schemas differ')
    return schema['schema_sha256']


def _sources():
    return {name: _sha(_read(REPO / name)) for name in SOURCES}


def _registration(source_sha256, binding_sha256):
    return dict(schema='teacher-cross-source-prereg-v1', base_commit=BASE_COMMIT,
        historical_source_commit=HIST_COMMIT, new_data_registration_commit=DATA_COMMIT,
        source_sha256=source_sha256, private_binding_sha256=binding_sha256,
        old_sources=['old', 'new'], old_groups=dict(GROUPS), old_candidates=dict(CANDIDATES),
        new_train_groups=59, new_dev_groups=62, max_candidates_per_group=256,
        comparisons=['raw_float32', 'baseline_normalized_float32'],
        match='full_ordered_numeric_group_with_exact_bytes_after_sha', conflict_priority=True,
        old_archive_seed_payload_decoded=False, max_attempts=1, job=dict(JOB),
        memory_initial_bytes=MEMORY_INITIAL, disk_floor_bytes=DISK_FLOOR,
        new_games=0, model_forward=False, training=False, independent_game_evaluation=False)


def prepare():
    """Bind metadata only; do not decode old candidate arrays or run a model."""
    expected_change = {'review/evidence/task-teacher-source-compat-v1-20261031/NOTE.zh-CN.md'}
    changed = set(filter(None, _git('diff', '--name-only').splitlines()))
    if (not _windows() or _git('rev-parse', 'HEAD') != BASE_COMMIT
            or changed != expected_change or _git('diff', '--cached', '--name-only')
            or REGISTRATION.exists() or PRIVATE.exists()):
        raise ValueError('Unexpected cross-source preparation state')
    if previous.physical_available() < MEMORY_INITIAL:
        raise MemoryError('Initial cross-source memory floor not met')
    deadline = time.monotonic() + 300
    guard = lambda: previous._guard(deadline)
    historical = _historical_metadata()
    train, dev, contract = transfer.load_trainable(guard=guard)
    if (len(train), len(dev), contract['registration_commit']) != (59, 62, DATA_COMMIT):
        raise ValueError('New source dataset differs')
    from learning.task_student_anchor_collection_v1 import sources
    context = sources.open_context(guard=guard)
    _mean, _scale, metadata = context.model_normalization()
    semantic_schema = _schema_identity(historical['schema_sha256'], metadata)
    binding = dict(schema='teacher-cross-source-private-binding-v1', salt=secrets.token_hex(32),
        historical=historical, train_sha256=contract['train_sha256'], dev_sha256=contract['dev_sha256'],
        baseline_sha256=context.task_sha256, schema_sha256=semantic_schema)
    raw = previous._encoded(binding)
    registration = _registration(_sources(), _sha(raw))
    PRIVATE.mkdir(parents=False, exist_ok=False)
    previous._write_new(PRIVATE / 'binding.local.json', binding)
    REGISTRATION.parent.mkdir(parents=True, exist_ok=True)
    previous._write_new(REGISTRATION, registration)
    print('Private source binding and public comparison registration prepared.')


def _registered_binding(head):
    if type(head) is not str or re.fullmatch(r'[0-9a-f]{40}', head) is None:
        raise ValueError('Invalid cross-source registration commit')
    parent = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', head + '^'],
                                     stderr=subprocess.DEVNULL, timeout=30).decode('ascii').strip()
    raw = subprocess.check_output(['git', '-C', str(REPO), 'show',
        head + ':' + REGISTRATION.relative_to(REPO).as_posix()], stderr=subprocess.DEVNULL, timeout=30)
    binding_raw = _read(PRIVATE / 'binding.local.json')
    reg = previous._document(raw)
    if parent != BASE_COMMIT or reg != _registration(_sources(), _sha(binding_raw)):
        raise ValueError('Cross-source registration or binding differs')
    binding = previous._document(binding_raw)
    if (binding.get('schema') != 'teacher-cross-source-private-binding-v1'
            or binding.get('historical') != _historical_metadata()):
        raise ValueError('Cross-source historical identity differs')
    return binding, _sha(binding_raw)


def _runtime_binding():
    head = _git('rev-parse', 'HEAD')
    if _git('status', '--porcelain') or _git('rev-parse', 'origin/main') != head:
        raise ValueError('Cross-source registration is not clean and published')
    binding, digest = _registered_binding(head)
    if _read(REGISTRATION) != subprocess.check_output(['git', '-C', str(REPO), 'show',
            head + ':' + REGISTRATION.relative_to(REPO).as_posix()], stderr=subprocess.DEVNULL, timeout=30):
        raise ValueError('Cross-source registration file differs')
    return head, binding, digest


def _result(raw, head):
    from .overlap import Match, _summary
    value = previous._document(raw)
    fields = {'schema', 'registration_commit', 'old_sources', 'old_groups', 'old_candidates',
              'new_train_groups', 'new_dev_groups', 'rows', 'summary', 'model_forward',
              'training', 'new_games', 'old_archive_seed_payload_decoded', 'passed'}
    if (set(value) != fields or value.get('schema') != 'teacher-cross-source-private-result-v1'
            or value.get('registration_commit') != head or value.get('old_sources') != ['old', 'new']
            or value.get('old_groups') != GROUPS or value.get('old_candidates') != CANDIDATES
            or value.get('new_train_groups') != 59 or value.get('new_dev_groups') != 62
            or value.get('model_forward') is not False or value.get('training') is not False
            or type(value.get('new_games')) is not int or value['new_games'] != 0
            or value.get('old_archive_seed_payload_decoded') is not False or value.get('passed') is not True
            or type(value.get('rows')) is not list or len(value['rows']) != 121):
        raise ValueError('Cross-source result fields differ')
    rows = []
    for i, row in enumerate(value['rows']):
        if (type(row) is not dict or set(row) != {'split', 'raw', 'normalized'}
                or row['split'] != ('train' if i < 59 else 'dev')
                or row['raw'] not in ('same', 'conflict', 'none')
                or row['normalized'] not in ('same', 'conflict', 'none')):
            raise ValueError('Cross-source result row differs')
        rows.append(Match(row['split'], row['raw'], row['normalized']))
    if value['summary'] != _summary(rows):
        raise ValueError('Cross-source result summary differs')
    return value


def child():
    head, binding, binding_sha = _runtime_binding()
    claim = previous._document(_read(PRIVATE / 'claim.local.json'))
    attempt = previous._document(_read(PRIVATE / 'attempt.local.json'))
    if (claim.get('schema') != 'teacher-cross-source-claim-v1' or claim.get('commit') != head
            or type(claim.get('attempt')) is not int or claim['attempt'] != 1
            or claim.get('freeze_sha256') != binding_sha or claim.get('retry_allowed') is not False
            or attempt != dict(schema='teacher-cross-source-attempt-v1', commit=head, retry_allowed=False)
            or (PRIVATE / 'failed.local.json').exists() or (PRIVATE / 'closed.local.json').exists()):
        raise ValueError('Cross-source child is not uniquely claimed')
    from learning.task_teacher_dev_deviation_v1.membership import ENV_KEY, verify_worker
    from learning.task_pair_fresh_multi_v1.entry import verify_live_owner
    import psutil
    owner = previous._document(_read(PRIVATE / 'owner.local.json'))
    verify_worker(os.environ.get(ENV_KEY), claim, binding_sha)
    verify_live_owner(owner, commit=head, binding_sha=binding_sha, current_pid=os.getpid(),
                      process=psutil.Process, executable=sys.executable, samefile=os.path.samefile,
                      now=time.monotonic())
    previous._write_new(PRIVATE / 'child-attempt.local.json', dict(
        schema='teacher-cross-source-child-attempt-v1', commit=head, retry_allowed=False))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    deadline = time.monotonic() + JOB['wall_seconds']
    guard = lambda: previous._guard(deadline)
    scanned = []
    try:
        from learning.task_student_anchor_collection_v1 import sources
        from . import archive, overlap
        train, dev, contract = transfer.load_trainable(guard=guard)
        if (len(train), len(dev), contract['registration_commit']) != (59, 62, DATA_COMMIT):
            raise ValueError('New source population differs')
        if (contract['train_sha256'] != binding.get('train_sha256')
                or contract['dev_sha256'] != binding.get('dev_sha256')):
            raise ValueError('New source identity differs')
        context = sources.open_context(guard=guard)
        mean, scale, metadata = context.model_normalization()
        if (context.task_sha256 != binding.get('baseline_sha256')
                or _schema_identity(binding['historical']['schema_sha256'], metadata) != binding.get('schema_sha256')):
            raise ValueError('Baseline or semantic schema differs')
        new_groups = [overlap.NewGroup(item.features, item.target_index, split)
                      for split, items in (('train', train), ('dev', dev)) for item in items]
        matcher = overlap.Matcher(new_groups, mean=mean, scale=scale, guard=guard)
        for key in ('old', 'new'):
            guard()
            def consume(_index, shared, candidates, label):
                matcher.consume(overlap.OldGroup(shared, candidates, label))
            receipt = archive.scan_archive(OLD_PATHS[key], expected_sha256=binding['historical']['archive_sha256'][key],
                expected_groups=GROUPS[key], expected_candidates=CANDIDATES[key], guard=guard, consume=consume)
            if receipt != dict(groups=GROUPS[key], candidates=CANDIDATES[key], verified_after_scan=True):
                raise ValueError('Training source scan did not close')
            scanned.append(key)
        if scanned != ['old', 'new']:
            raise ValueError('Training source scan order differs')
        comparison = matcher.finish()
        if comparison.old_groups != sum(GROUPS.values()):
            raise ValueError('Training source group count differs')
        rows = [dict(split=row.split, raw=row.raw, normalized=row.normalized) for row in comparison.rows]
        result = dict(schema='teacher-cross-source-private-result-v1', registration_commit=head,
            old_sources=['old', 'new'], old_groups=dict(GROUPS), old_candidates=dict(CANDIDATES),
            new_train_groups=59, new_dev_groups=62, rows=rows, summary=comparison.summary,
            model_forward=False, training=False, new_games=0, old_archive_seed_payload_decoded=False, passed=True)
        _result(previous._encoded(result), head)
        guard()
        previous._write_new(PRIVATE / 'worker-result.local.json', result)
    except BaseException as exc:
        previous._write_new(PRIVATE / 'worker-failed.local.json', dict(
            category=type(exc).__name__, completed_archives=len(scanned), retry_allowed=False))
        raise


def run():
    if not _windows() or previous.physical_available() < MEMORY_INITIAL:
        raise MemoryError('Cross-source initial memory floor not met')
    if shutil.disk_usage(PRIVATE).free < DISK_FLOOR:
        raise OSError('Cross-source disk floor not met')
    head, _binding, binding_sha = _runtime_binding()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    from learning.task_teacher_dev_deviation_v1.membership import token_hash
    from learning.task_pair_fresh_multi_v1.entry import owner_marker
    import psutil
    deadline = time.monotonic() + JOB['wall_seconds'] + 2 * JOB['cleanup_seconds']
    previous._write_new(PRIVATE / 'attempt.local.json', dict(
        schema='teacher-cross-source-attempt-v1', commit=head, retry_allowed=False))
    try:
        proc = psutil.Process(os.getpid())
        owner = previous._document(owner_marker(commit=head, binding_sha=binding_sha, pid=os.getpid(),
            created_us=int(proc.create_time() * 1_000_000), deadline=deadline))
        previous._write_new(PRIVATE / 'owner.local.json', owner)
        job_name = 'Local\\TeacherDeviation-' + secrets.token_hex(16)
        previous._write_new(PRIVATE / 'claim.local.json', dict(schema='teacher-cross-source-claim-v1',
            commit=head, attempt=1, freeze_sha256=binding_sha, job_token_sha256=token_hash(job_name), retry_allowed=False))
        command = [sys.executable, '-B', '-m', 'learning.task_teacher_cross_source_v1.experiment', 'child']
        job = supervisor._supervise_owned(command, str(REPO / 'kaggriculture'), JOB, job_name)
        previous._write_new(PRIVATE / 'job.local.json', job)
        if not transfer._job_ok(job):
            raise RuntimeError('Cross-source owned Job did not close cleanly')
        result_raw = _read(PRIVATE / 'worker-result.local.json')
        result = _result(result_raw, head)
        previous._write_new(PRIVATE / 'closed.local.json', dict(schema='teacher-cross-source-closed-v1',
            registration_commit=head, worker_sha256=_sha(result_raw),
            job_sha256=_sha(_read(PRIVATE / 'job.local.json')),
            attempt_sha256=_sha(_read(PRIVATE / 'attempt.local.json')),
            owner_sha256=_sha(_read(PRIVATE / 'owner.local.json')),
            claim_sha256=_sha(_read(PRIVATE / 'claim.local.json')),
            child_attempt_sha256=_sha(_read(PRIVATE / 'child-attempt.local.json')),
            binding_sha256=binding_sha, retry_allowed=False))
        print(json.dumps(dict(status='completed', summary=result['summary']), sort_keys=True), flush=True)
    except BaseException as exc:
        previous._write_new(PRIVATE / 'failed.local.json', dict(
            status='failed', category=type(exc).__name__, retry_allowed=False))
        raise


def load_completed():
    if (PRIVATE / 'failed.local.json').exists() or (PRIVATE / 'worker-failed.local.json').exists():
        raise ValueError('Failed cross-source attempt cannot be consumed')
    closed = previous._document(_read(PRIVATE / 'closed.local.json'))
    head = closed.get('registration_commit')
    _binding, binding_sha = _registered_binding(head)
    if (closed.get('schema') != 'teacher-cross-source-closed-v1' or closed.get('retry_allowed') is not False
            or closed.get('binding_sha256') != binding_sha):
        raise ValueError('Cross-source seal differs')
    for name, key in (('worker-result.local.json', 'worker_sha256'), ('job.local.json', 'job_sha256'),
                      ('attempt.local.json', 'attempt_sha256'), ('owner.local.json', 'owner_sha256'),
                      ('claim.local.json', 'claim_sha256'), ('child-attempt.local.json', 'child_attempt_sha256')):
        if _sha(_read(PRIVATE / name)) != closed.get(key):
            raise ValueError('Cross-source receipt identity differs')
    if not transfer._job_ok(previous._document(_read(PRIVATE / 'job.local.json'))):
        raise ValueError('Cross-source Job receipt differs')
    if previous._document(_read(PRIVATE / 'attempt.local.json')) != dict(
            schema='teacher-cross-source-attempt-v1', commit=head, retry_allowed=False):
        raise ValueError('Cross-source attempt differs')
    claim = previous._document(_read(PRIVATE / 'claim.local.json'))
    marker = previous._document(_read(PRIVATE / 'child-attempt.local.json'))
    owner = previous._document(_read(PRIVATE / 'owner.local.json'))
    if (claim.get('schema') != 'teacher-cross-source-claim-v1' or claim.get('commit') != head
            or type(claim.get('attempt')) is not int or claim['attempt'] != 1
            or claim.get('freeze_sha256') != binding_sha or claim.get('retry_allowed') is not False
            or type(claim.get('job_token_sha256')) is not str
            or re.fullmatch(r'[0-9a-f]{64}', claim['job_token_sha256']) is None
            or marker != dict(schema='teacher-cross-source-child-attempt-v1', commit=head, retry_allowed=False)
            or type(owner) is not dict or set(owner) != {'schema', 'registration_commit', 'binding_sha256',
                                                         'pid', 'created_us', 'deadline'}
            or owner.get('schema') != 'task-pair-fresh-multi-owner-v1'
            or owner.get('registration_commit') != head or owner.get('binding_sha256') != binding_sha
            or type(owner.get('pid')) is not int or owner['pid'] <= 0
            or type(owner.get('created_us')) is not int or owner['created_us'] <= 0
            or type(owner.get('deadline')) not in (int, float)):
        raise ValueError('Cross-source claim differs')
    return _result(_read(PRIVATE / 'worker-result.local.json'), head)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'child'))
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'child': child}[args.command]()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        kind = 'memory' if isinstance(exc, MemoryError) else 'timeout' if isinstance(exc, TimeoutError) else 'io' if isinstance(exc, OSError) else 'invalid_or_runtime'
        print(json.dumps(dict(status='failed', category=kind)), file=sys.stderr)
        raise SystemExit(1) from None
