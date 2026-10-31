"""Privately bind old models and already-consumed seeds for a new panel.

This metadata-only step never opens a replay or a model and never starts a
game. Its output, selected identities and salt stay outside the public repo.
"""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent / 'kaggriculture'
PRIVATE = REPO.parent / 'private'
BASE = 'c70bdf406a9eb4d9a3de66d37fee65bc102a9e63'
STAGE = 'task-pair-market-consumed-v1-20261031'
OLD_BINDING = 'task-equivalent-bc-v1-binding-v2.local.json'
OLD_MANIFEST = 'task-pair-round2-compare-v1-20261031/comparison-manifest.local.json'
FAILED_MANIFEST = 'task-pair-historical-compare-v1-20261030/comparison-manifest.local.json'
CHOICE = 'task-choice-v1-20261023/registration.json'
CHOICE_PUBLIC = 'review/evidence/task-choice-v1-20261023/registration.json'
PROTECTED = (
    ('task-teacher-merged-eval-v1-20261028', '692b9c7cd640c31bfb319e7ae434c6fa7176ba43'),
    ('task-teacher-merged-eval-v2-20261029', 'e2d96abe357dd2a59f9cc23a23953bca8e967e7c'),
    ('task-teacher-merged-eval-v3-smoke-20261030', '59d6f302a72e66cbd435b1933de94b723e3c05c0'),
    ('task-teacher-merged-eval-v3-formal-20261030', 'e64f3bad589268a7b1b7c57844a602b595cca2d4'),
)
MAX_FILE = 16 * 1024**2
MAX_PUBLIC_BYTES = 256 * 1024**2
HEX = re.compile(r'[0-9a-f]{64}\Z')
SKIP = frozenset(('count', 'counts', 'size', 'length', 'bytes', 'hash', 'hashes',
                  'sha', 'sha256', 'digest', 'checksum', 'score', 'profit', 'reward',
                  'index', 'indices', 'min', 'max', 'range', 'probability',
                  'mean', 'median', 'variance', 'limit', 'budget', 'threshold', 'seconds',
                  'cash', 'price', 'revenue', 'sales', 'amount', 'units', 'turn', 'step'))
RISK = frozenset(('test', 'holdout', 'reserved'))


def require(value):
    if not value:
        raise ValueError('Seed preparation identity or scope differs')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(items):
    out = {}
    for key, value in items:
        require(key not in out)
        out[key] = value
    return out


def _bad(_):
    raise ValueError('Nonfinite JSON')


def decode(raw):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_FILE)
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_pairs, parse_constant=_bad)


def read(path, limit=MAX_FILE):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink()
            and not (getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)
            and path.stat().st_size <= limit)
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    require(0 < len(data) <= limit)
    return data


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], stderr=subprocess.DEVNULL)


def _tokens(value):
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', str(value)).casefold()
    return text, set(re.split(r'[^a-z0-9]+', text))


def _skip_field(label, tokens):
    return (bool(tokens & SKIP) or 'excluded' in tokens
            or label in ('number_of_seeds', 'num_seeds', 'n_seeds'))


def _by_seed(value):
    """In a statistics map, numeric keys are seeds; metric values are not."""
    found = set()
    if type(value) is list:
        for item in value:
            if type(item) is dict:
                found.update(reserved_values(item, risk=True))
            else:
                found.update(_seed_numbers(item))
        return found
    require(type(value) is dict)
    for key, item in value.items():
        label, _ = _tokens(key)
        if label.isdecimal():
            if 0 <= int(label) < 2**32:
                found.add(int(label))
        elif re.fullmatch(r'(?:s[0-9]+|alias[_-]?[a-z0-9]+)', label):
            if type(item) in (int, float, str):
                found.update(_seed_numbers(item))
            elif type(item) is dict:
                found.update(reserved_values(item, risk=True))
    return found


def _seed_numbers(value):
    """Conservatively extract scalar and numeric-key uint32 seed values."""
    found = set()
    if type(value) is dict:
        for key, item in value.items():
            label, tokens = _tokens(key)
            if _skip_field(label, tokens):
                continue
            if label in ('by_seed', 'per_seed'):
                found.update(_by_seed(item))
                continue
            if label.isdecimal() and 0 <= int(label) < 2**32:
                found.add(int(label))
                found.update(_seed_numbers(item))
            else:
                # Within an explicit seeds container, an unknown grouping key
                # might be an opponent or pool. Treat its numeric descendants
                # as reserved rather than silently overlooking that group.
                found.update(_seed_numbers(item))
    elif type(value) is list:
        for item in value:
            found.update(_seed_numbers(item))
    elif type(value) is int and 0 <= value < 2**32:
        found.add(value)
    elif type(value) is float:
        require(value.is_integer())
        if 0 <= value < 2**32:
            found.add(int(value))
    elif type(value) is str and len(value) <= 128:
        if re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', value):
            try:
                number = Decimal(value)
                require(number.is_finite() and number == number.to_integral_value())
                if 0 <= number < 2**32:
                    found.add(int(number))
            except InvalidOperation as exc:
                raise ValueError('Invalid seed metadata') from exc
    return found


def reserved_values(value, *, risk=False, seed_context=False):
    """Find seed fields under explicit test/holdout/reserved contexts.

    Excluded-seed inventories are not themselves future evaluation seeds.
    Generic unlabelled private documents remain outside this scanner's scope.
    """
    found = set()
    if type(value) is dict:
        for key, item in value.items():
            label, tokens = _tokens(key)
            if _skip_field(label, tokens):
                continue
            next_risk = risk or bool(tokens & RISK)
            next_seed = seed_context or 'seed' in label
            if next_risk and label in ('by_seed', 'per_seed'):
                found.update(_by_seed(item))
                continue
            if next_risk and next_seed:
                found.update(_seed_numbers(item))
            else:
                found.update(reserved_values(item, risk=next_risk, seed_context=next_seed))
    elif type(value) is list:
        for item in value:
            found.update(reserved_values(item, risk=risk, seed_context=seed_context))
    return found


def _public_reserved(commit):
    require(type(commit) is str and re.fullmatch(r'[0-9a-f]{40}', commit))
    listing = git('ls-tree', '-r', '-z', '--name-only', commit)
    names = sorted(x.decode('utf8') for x in listing.split(b'\0') if x.endswith(b'.json'))
    require(names and len(names) <= 10000)
    process = subprocess.Popen(['git', '-C', str(REPO), 'cat-file', '--batch'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    result, total = set(), 0
    try:
        for name in names:
            process.stdin.write((commit + ':' + name + '\n').encode('utf8'))
            process.stdin.flush()
            header = process.stdout.readline().split()
            require(len(header) == 3 and header[1] == b'blob')
            size = int(header[2]); total += size
            require(0 < size <= MAX_FILE and total <= MAX_PUBLIC_BYTES)
            raw = process.stdout.read(size)
            require(len(raw) == size and process.stdout.read(1) == b'\n')
            result.update(reserved_values(decode(raw)))
        process.stdin.close()
        require(process.wait() == 0)
    except BaseException:
        process.kill(); process.wait()
        raise
    return result, len(names), total


def select(train, validation, old, reserved):
    require(type(train) is list and type(validation) is list and type(old) is set
            and type(reserved) is set and all(type(x) is int and 0 <= x < 2**32
                                               for x in train + validation + list(old) + list(reserved)))
    require(len(set(train)) == len(train) and len(set(validation)) == len(validation)
            and not set(train) & set(validation))
    prior_train = [seed for seed in train if seed not in old][:2]
    prior_dev = [seed for seed in validation if seed not in old and seed not in prior_train][:1]
    require(len(prior_train) == 2 and len(prior_dev) == 1)
    forbidden = old | reserved | set(prior_train) | set(prior_dev)
    chosen = [seed for seed in train if seed not in forbidden][:2]
    require(len(chosen) == 2 and len(set(chosen)) == 2)
    return chosen


def verify_consumed(acceptance, ledger, selected):
    """Require six passed historical reconstruction coordinates per seed."""
    require(type(acceptance) is dict and acceptance.get('stage_execution_passed') is True
            and acceptance.get('original_demonstrations_reconstructed') == 60
            and type(ledger) is dict and ledger.get('passed') is True
            and ledger.get('games') == 60 and ledger.get('failures') == []
            and type(ledger.get('rows')) is list and len(ledger['rows']) == 60
            and type(selected) is list and len(selected) == 2 and len(set(selected)) == 2)
    expected = {(seed, opponent, side) for seed in selected
                for opponent in ('v1', 'market', 'round2') for side in (0, 1)}
    seen = set()
    for row in ledger['rows']:
        require(type(row) is dict)
        if row.get('seed') not in selected:
            continue
        key = (row.get('seed'), row.get('opponent'), row.get('side'))
        audit = row.get('audit')
        require(type(row.get('seed')) is int and type(row.get('side')) is int
                and key in expected and key not in seen and row.get('split') == 'train'
                and row.get('passed') is True and row.get('original_teacher_actions_checked') == 719
                and row.get('reconstructed_actor_actions') == row.get('reproduced_rival_actions') == 719
                and type(audit) is dict and audit.get('passed') is True
                and audit.get('official_transitions_reexecuted') == 719
                and audit.get('both_public_and_private_states_match_each_step') is True
                and audit.get('inventory_order_checked_each_step') is True)
        seen.add(key)
    require(seen == expected)


def collect(old_binding_sha, salt):
    """Recompute the private binding from authenticated metadata only."""
    require(type(old_binding_sha) is str and HEX.fullmatch(old_binding_sha) is not None
            and type(salt) is str and HEX.fullmatch(salt) is not None)
    git('merge-base', '--is-ancestor', BASE, 'HEAD')
    original = read(PRIVATE / OLD_BINDING, 65536)
    require(sha(original) == old_binding_sha)
    binding = decode(original)
    old_raw = read(GAME / 'artifacts' / OLD_MANIFEST, 8 * 1024**2)
    failed_raw = read(GAME / 'artifacts' / FAILED_MANIFEST, 8 * 1024**2)
    require(sha(old_raw) == binding['old_manifest_sha256'])
    old, failed = decode(old_raw), decode(failed_raw)
    for key in ('models', 'crop', 'opponents', 'worker_sha256'):
        require(old[key] == failed[key])
    require(set(old['seeds']) == set(failed['seeds']) == {'S02', 'S03'}
            and set(old['seeds'].values()) == set(failed['seeds'].values()))
    choice_raw = read(GAME / 'artifacts' / CHOICE, 8 * 1024**2)
    require(choice_raw == git('show', BASE + ':' + CHOICE_PUBLIC))
    choice = decode(choice_raw)
    source_dir = GAME / 'artifacts' / 'task-choice-v1-20261023'
    acceptance_raw = read(source_dir / 'acceptance.json', 8 * 1024**2)
    require(sha(acceptance_raw) == binding['acceptance_sha256']
            and acceptance_raw == git('show', BASE + ':review/evidence/task-choice-v1-20261023/acceptance.json'))
    acceptance = decode(acceptance_raw)
    ledger_raw = read(source_dir / 'reconstruction-ledger.json', 8 * 1024**2)
    require(ledger_raw == git('show', BASE + ':review/evidence/task-choice-v1-20261023/reconstruction-ledger.json')
            and acceptance['evidence_sha256']['registration.json'] == sha(choice_raw)
            and acceptance['evidence_sha256']['reconstruction-ledger.json'] == sha(ledger_raw))
    ledger = decode(ledger_raw)
    reserved, public_count, public_bytes = _public_reserved(BASE)
    protected_hashes = {}
    for name, commit in PROTECTED:
        git('merge-base', '--is-ancestor', commit, BASE)
        local_reg = read(GAME / 'artifacts' / name / 'registration.json')
        registered = git('show', commit + ':review/evidence/' + name + '/registration.json')
        require(local_reg == registered)
        protected_raw = read(GAME / 'artifacts' / name / 'private-seeds.local.json', 16384)
        cfg, protected = decode(registered), decode(protected_raw)
        require(cfg.get('seed_document_sha256') == sha(protected_raw)
                and type(protected.get('seeds')) is list and len(protected['seeds']) == 4
                and len(set(protected['seeds'])) == 4
                and all(type(x) is int and 0 <= x < 2**32 for x in protected['seeds'])
                and type(protected.get('salt')) is str and HEX.fullmatch(protected['salt']))
        reserved.update(protected['seeds'])
        protected_hashes[name] = sha(protected_raw)
    seeds = select(choice['train_seeds'], choice['validation_seeds'],
                   set(old['seeds'].values()) | set(failed['seeds'].values()), reserved)
    verify_consumed(acceptance, ledger, seeds)
    return dict(schema='task-pair-market-consumed-binding-v1', base_commit=BASE,
        salt=salt, seeds={'S02': seeds[0], 'S03': seeds[1]},
        old_binding_sha256=old_binding_sha, old_manifest_sha256=sha(old_raw),
        failed_manifest_sha256=sha(failed_raw), choice_registration_sha256=sha(choice_raw),
        choice_acceptance_sha256=sha(acceptance_raw), choice_ledger_sha256=sha(ledger_raw),
        protected_document_sha256=protected_hashes, protected_public_files=public_count,
        protected_public_bytes=public_bytes, models=old['models'], crop=old['crop'],
        opponents=old['opponents'], worker_sha256=old['worker_sha256'],
        prepare_source_sha256=sha(read(Path(__file__))))


def prepare(old_binding_sha):
    require(git('rev-parse', 'HEAD').decode().strip() == BASE)
    require(not git('diff', '--name-only', '-z').strip(b'\0')
            and not git('diff', '--cached', '--name-only', '-z').strip(b'\0'))
    untracked = [x.decode('utf8') for x in git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if x]
    require(untracked and all(name.startswith('kaggriculture/learning/task_pair_market_consumed_v1/')
                              for name in untracked))
    document = collect(old_binding_sha, os.urandom(32).hex())
    directory = PRIVATE / STAGE
    directory.mkdir(exist_ok=False)
    raw = (json.dumps(document, separators=(',', ':'), allow_nan=False) + '\n').encode()
    with (directory / 'binding.local.json').open('xb') as stream:
        require(stream.write(raw) == len(raw))
    with (directory / 'binding-sha256.local.txt').open('xb') as stream:
        require(stream.write((sha(raw) + '\n').encode()) == 65)
    print(json.dumps({'status': 'bound', 'seed_roles': 2, 'public_metadata_files': document['protected_public_files'],
                      'private_documents': len(PROTECTED)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-binding-sha256', required=True)
    args = parser.parse_args()
    try:
        prepare(args.old_binding_sha256)
    except BaseException:
        print(json.dumps({'status': 'failed', 'phase': 'private_binding'}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
