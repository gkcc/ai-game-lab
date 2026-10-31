"""Privately freeze an inventory and four fresh exploratory seed roles.

This standard-library-only preparation starts no game or training.  Public Git
JSON at one base commit, recognized local JSON metadata, and authenticated
historical private bindings are inventoried before any random draw.  The
private immutable snapshot and binding are not a third-party timestamp, and
unlisted non-JSON, deleted or external history cannot be ruled out.
"""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
from types import ModuleType


REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent / 'kaggriculture'
ARTIFACTS = GAME / 'artifacts'
PRIVATE_ROOT = REPO.parent / 'private'
PRIVATE = PRIVATE_ROOT / 'task-pair-fresh-multi-v1-20261031'
BASE = '8f6448e641db98b521dc5336d3b0b2d9217739d4'
PREFIX = 'kaggriculture/learning/task_pair_fresh_multi_v1/'
MARKET_PREP = 'kaggriculture/learning/task_pair_market_consumed_v1/prepare.py'
MARKET_PRIVATE = PRIVATE_ROOT / 'task-pair-market-consumed-v1-20261031'
MARKET_STAGE = ARTIFACTS / 'task-pair-historical-compare-v1-consumed-market-20261031'
MARKET_COMMIT = 'e5b613d8b9eab5daa0abeb7d3d160e0f20f877cd'
MARKET_PLAN = 'review/evidence/market-teacher-v1-20261021/development-plan.json'
MARKET_FREEZE = 'review/evidence/market-teacher-v1-20261021/development-freeze.json'
MARKET_ACCEPTANCE = 'review/evidence/market-teacher-v1-20261021/acceptance.json'
CHOICE = 'review/evidence/task-choice-v1-20261023/registration.json'
REQUIRED_SOURCES = tuple(PREFIX + name for name in (
    '__init__.py', 'prepare.py', 'test_prepare.py', 'entry.py', 'test_entry.py',
    'coordinator.py', 'test_coordinator.py', 'projection.py', 'test_projection.py'))
HEX = re.compile(r'[0-9a-f]{64}\Z')
INTEGER_TOKEN = re.compile(rb'(?<![0-9])[0-9]+(?![0-9])')
EXPONENT_TOKEN = re.compile(rb'(?<![0-9])(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][+-]?[0-9]+(?![0-9])')
NUMERIC_TEXT = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
LOCAL_NAMES = re.compile(r'registration|manifest|ledger|config|result|summary|seed|attempt|reservation|dispatch|debit', re.I)
MAX_FILE = 16 * 1024**2
MAX_TOTAL = 256 * 1024**2
MAX_SNAPSHOT = 32 * 1024**2
MAX_SOURCES = 48
MAX_FILES = 30000
MAX_DRAWS = 1024
DOMAIN = b'task-pair-fresh-multi-v1-seed\x00'
ROLES = ('F01', 'F02', 'F03', 'F04')
COHORTS = {'C01': {'S02': 'F01', 'S03': 'F02'},
           'C02': {'S02': 'F03', 'S03': 'F04'}}


def require(condition):
    if not condition:
        raise ValueError('Fresh multi-opponent preparation check failed')


def sha(raw):
    require(type(raw) is bytes)
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result)
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError('Nonfinite JSON')


def decode(raw):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_SNAPSHOT)
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_pairs,
                      parse_float=Decimal, parse_constant=_nonfinite)


def raw_numbers(raw):
    """Conservatively inspect integer and exponent tokens in arbitrary bytes."""
    require(type(raw) is bytes)
    found = set()
    for match in INTEGER_TOKEN.finditer(raw):
        digits = match.group().lstrip(b'0') or b'0'
        if len(digits) <= 10:
            number = int(digits)
            if number < 2**32:
                found.add(number)
    for match in EXPONENT_TOKEN.finditer(raw):
        token = match.group().decode('ascii')
        require(len(token) <= 128)
        number = _numeric(token)
        if number is not None:
            found.add(number)
        # A dot can be a decimal point or a filename separator. Exclude both
        # interpretations rather than silently missing e.g. "seed.1e3.json".
        mantissa, exponent = re.split('[eE]', token, maxsplit=1)
        if '.' in mantissa and mantissa.rsplit('.', 1)[1]:
            alternate = mantissa.rsplit('.', 1)[1] + 'e' + exponent
            number = _numeric(alternate)
            if number is not None:
                found.add(number)
    return found


def _numeric(value):
    """Return a uint32 for a numeric value or pure numeric string, else None."""
    if type(value) is int:
        number = value
    elif type(value) is Decimal:
        number = value
    elif type(value) is str and NUMERIC_TEXT.fullmatch(value):
        # Long numeric strings can encode a small value using zeros or exponents.
        require(len(value) <= 128)
        text = value.casefold()
        if 'e' in text:
            mantissa, exponent = text.split('e', 1)
            magnitude = exponent.lstrip('+-').lstrip('0') or '0'
            if len(magnitude) > 4 or int(magnitude) > 1000:
                # With a <=128-character mantissa, nonzero values at these
                # exponents are either beyond uint32 or nonintegral.
                digits = [char for char in mantissa if char.isdecimal()]
                require(digits)
                return 0 if all(char == '0' for char in digits) else None
        try:
            number = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError('Invalid numeric metadata') from exc
    else:
        return None
    if isinstance(number, Decimal):
        require(number.is_finite())
        if number < 0 or number >= 2**32 or number != number.to_integral_value():
            return None
    elif number < 0 or number >= 2**32:
        return None
    return int(number)


def structured_numbers(value):
    """Conservative recursive scalar and mapping-key scan; no role heuristics."""
    stack, found = [value], set()
    while stack:
        item = stack.pop()
        if type(item) is dict:
            for key, child in item.items():
                require(type(key) is str)
                stack.append(key)
                stack.append(child)
        elif type(item) is list:
            stack.extend(item)
        else:
            if type(item) is str:
                # JSON escapes may hide numeric tokens from raw-byte scans.
                found.update(raw_numbers(item.encode('utf-8')))
            number = _numeric(item)
            if number is not None:
                found.add(number)
    return found


def scan_json(raw, relative):
    require(type(relative) is str and relative and '\n' not in relative and '\r' not in relative)
    return raw_numbers(raw) | raw_numbers(relative.encode('utf-8')) | structured_numbers(decode(raw))


def _git(*args):
    with open(os.devnull, 'rb') as null:
        result = subprocess.run(['git', '-C', str(REPO), *args], stdin=null,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(result.returncode == 0 and len(result.stdout) <= MAX_TOTAL)
    return result.stdout


def _blob(relative, commit=BASE):
    require(type(relative) is str and relative and '\n' not in relative and '\r' not in relative)
    return _git('show', commit + ':' + relative)


def _inside(root, path):
    root, path = Path(root).absolute(), Path(path).absolute()
    require(root.is_absolute() and path != root and path.is_relative_to(root))
    for index in range(len(path.relative_to(root).parts) + 1):
        item = root.joinpath(*path.relative_to(root).parts[:index])
        info = item.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, 'st_file_attributes', 0) & 0x400))
    require(root.resolve(strict=True) == root and path.resolve(strict=True) == path)
    return path


def _read(root, path, maximum=MAX_FILE):
    path = _inside(root, path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum)
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    after = path.lstat()
    require(len(raw) == before.st_size == after.st_size <= maximum
            and before.st_mtime_ns == after.st_mtime_ns and _inside(root, path) == path)
    return raw


def _tracked_names():
    listing = _git('ls-tree', '-r', '-z', '--name-only', BASE)
    names = sorted(part.decode('utf-8') for part in listing.split(b'\0') if part.lower().endswith(b'.json'))
    require(0 < len(names) <= MAX_FILES and len(names) == len(set(names))
            and all('\n' not in name and '\r' not in name for name in names))
    return names


def _public_inventory():
    """Read the exact base-commit JSON blobs; no checkout data is substituted."""
    names = _tracked_names()
    process = subprocess.Popen(['git', '-C', str(REPO), 'cat-file', '--batch'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    found, files, total = set(), [], 0
    try:
        for name in names:
            process.stdin.write((BASE + ':' + name + '\n').encode('utf-8'))
            process.stdin.flush()
            header = process.stdout.readline().split()
            require(len(header) == 3 and header[1] == b'blob')
            size = int(header[2]); total += size
            require(0 < size <= MAX_FILE and total <= MAX_TOTAL)
            raw = process.stdout.read(size)
            require(len(raw) == size and process.stdout.read(1) == b'\n')
            found.update(scan_json(raw, name))
            files.append(dict(path=name, sha256=sha(raw), bytes=size))
        process.stdin.close()
        require(process.wait(timeout=60) == 0)
    except BaseException:
        process.kill()
        process.wait()
        raise
    return found, files, total


def include_local_metadata(relative):
    """Skip raw replay.json, but retain replay-named result and attempt files."""
    path = PurePosixPath(relative)
    require(type(relative) is str and path.as_posix() == relative and not path.is_absolute()
            and all(part not in ('', '.', '..') for part in path.parts))
    name = path.name
    return (path.suffix.lower() == '.json' and name.lower() != 'replay.json'
            and bool(LOCAL_NAMES.search(name) or name.lower().startswith('replay')
                     or any(part.casefold() == 'job-inputs' for part in path.parts)))


def _local_paths():
    """Enumerate recognized artifact metadata; never follow reparse points."""
    require(ARTIFACTS.is_dir() and not ARTIFACTS.is_symlink()
            and ARTIFACTS.resolve(strict=True) == ARTIFACTS)
    paths = []
    def fail(error):
        raise OSError('Artifact metadata enumeration failed') from error
    for directory, dirs, names in os.walk(ARTIFACTS, topdown=True, followlinks=False, onerror=fail):
        base = Path(directory)
        for name in dirs:
            path = base / name
            info = path.lstat()
            require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
                    and not (getattr(info, 'st_file_attributes', 0) & 0x400))
        for name in names:
            path = base / name
            if not include_local_metadata(path.relative_to(ARTIFACTS).as_posix()):
                continue
            paths.append(path)
            require(len(paths) <= MAX_FILES)
    return sorted(paths, key=lambda p: p.relative_to(ARTIFACTS).as_posix())


def _local_inventory(initial_total):
    found, files, total = set(), [], initial_total
    for path in _local_paths():
        raw = _read(ARTIFACTS, path)
        total += len(raw)
        require(total <= MAX_TOTAL)
        relative = path.relative_to(ARTIFACTS).as_posix()
        found.update(scan_json(raw, relative))
        files.append(dict(path=relative, sha256=sha(raw), bytes=len(raw)))
    require(files)
    return found, files, total


def _json_file(root, path, maximum=MAX_FILE):
    raw = _read(root, path, maximum)
    value = decode(raw)
    require(type(value) is dict)
    return raw, value


def _seed_list(value, *, count=None):
    require(type(value) is list and (count is None or len(value) == count)
            and value and all(type(item) is int and 0 <= item < 2**32 for item in value)
            and len(set(value)) == len(value))
    return set(value)


def _market_module():
    raw = _read(REPO, REPO / MARKET_PREP)
    require(raw == _blob(MARKET_PREP))
    module = ModuleType('_fresh_multi_market_prepare')
    module.__file__ = str(REPO / MARKET_PREP)
    exec(compile(raw, module.__file__, 'exec', dont_inherit=True), module.__dict__)
    return module


def load_verified_market_binding():
    """Recompute prior private market binding, without exposing its contents."""
    module = _market_module()
    raw, market = _json_file(PRIVATE_ROOT, MARKET_PRIVATE / 'binding.local.json', 65536)
    reference = _read(PRIVATE_ROOT, MARKET_PRIVATE / 'binding-sha256.local.txt', 256)
    require(reference.decode('ascii').strip() == sha(raw)
            and market.get('schema') == 'task-pair-market-consumed-binding-v1'
            and module.collect(market['old_binding_sha256'], market['salt']) == market)
    attempt_raw, attempt = _json_file(PRIVATE_ROOT, MARKET_PRIVATE / 'attempt.local.json', 65536)
    del attempt_raw
    require(attempt.get('kind') == 'attempt' and attempt.get('registration_commit') == MARKET_COMMIT
            and attempt.get('binding_sha256') == sha(raw) and attempt.get('retry_allowed') is False
            and _read(ARTIFACTS, MARKET_STAGE / 'manifest.local.json', 65536) == raw)
    return raw, market


def known_history():
    """Authenticate known reservations and return private roles and identities."""
    module = _market_module()
    market_raw, market = load_verified_market_binding()
    choice_blob = _blob(CHOICE)
    choice_local = _read(ARTIFACTS, ARTIFACTS / 'task-choice-v1-20261023' / 'registration.json')
    require(choice_local == choice_blob)
    choice = decode(choice_blob)
    require(type(choice) is dict)
    train = _seed_list(choice['train_seeds'], count=8)
    validation = _seed_list(choice['validation_seeds'], count=2)
    require(not train & validation)

    plan_raw, freeze_raw, acceptance_raw = (_blob(name) for name in (MARKET_PLAN, MARKET_FREEZE, MARKET_ACCEPTANCE))
    require(all(_read(REPO, REPO / name) == raw for name, raw in
                ((MARKET_PLAN, plan_raw), (MARKET_FREEZE, freeze_raw), (MARKET_ACCEPTANCE, acceptance_raw))))
    plan, freeze, acceptance = (decode(raw) for raw in (plan_raw, freeze_raw, acceptance_raw))
    require(type(plan) is dict and type(freeze) is dict and type(acceptance) is dict
            and plan.get('games') == 192 and freeze.get('registration_sha256') == sha(plan_raw)
            and acceptance.get('formal_test_games') == 0
            and _read(ARTIFACTS, ARTIFACTS / 'market-teacher-v1-20261021' / 'development-plan.json') == plan_raw)
    market_holdout = _seed_list(plan['seeds'], count=12)
    market_development = _seed_list(plan['development_seeds'], count=2)
    require(not market_holdout & market_development)

    protected = set()
    protected_files = []
    for folder, commit in module.PROTECTED:
        _git('merge-base', '--is-ancestor', commit, BASE)
        rel = 'review/evidence/' + folder + '/registration.json'
        registration_raw = _blob(rel, commit)
        registration_local = ARTIFACTS / folder / 'registration.json'
        require(_read(ARTIFACTS, registration_local) == registration_raw)
        registration = decode(registration_raw)
        private_path = ARTIFACTS / folder / 'private-seeds.local.json'
        private_raw, private = _json_file(ARTIFACTS, private_path, 16384)
        require(type(registration) is dict and registration.get('seed_document_sha256') == sha(private_raw)
                and type(private.get('salt')) is str and HEX.fullmatch(private['salt']))
        seeds = _seed_list(private.get('seeds'), count=4)
        protected.update(seeds)
        protected_files.append(dict(path=private_path.relative_to(ARTIFACTS).as_posix(),
                                    sha256=sha(private_raw), bytes=len(private_raw)))
    require(len(protected_files) == 4)

    old_path = PRIVATE_ROOT / module.OLD_BINDING
    old_raw, old_binding = _json_file(PRIVATE_ROOT, old_path, 65536)
    round_raw, round_manifest = _json_file(ARTIFACTS, ARTIFACTS / module.OLD_MANIFEST)
    failed_raw, failed_manifest = _json_file(ARTIFACTS, ARTIFACTS / module.FAILED_MANIFEST)
    require(sha(old_raw) == market['old_binding_sha256']
            and sha(round_raw) == old_binding['old_manifest_sha256'] == market['old_manifest_sha256']
            and sha(failed_raw) == market['failed_manifest_sha256'])
    def mapped(manifest):
        mapping = manifest.get('seeds')
        require(type(mapping) is dict and set(mapping) == {'S02', 'S03'})
        return _seed_list(list(mapping.values()), count=2)
    market_roles = mapped(market)
    round_roles = mapped(round_manifest)
    failed_roles = mapped(failed_manifest)
    require(round_roles == failed_roles)

    private_paths = (old_path, MARKET_PRIVATE / 'binding.local.json',
                     MARKET_PRIVATE / 'binding-sha256.local.txt', MARKET_PRIVATE / 'attempt.local.json')
    private_sources = []
    extra_values = set()
    for path in private_paths:
        raw = _read(PRIVATE_ROOT, path, 65536)
        rel = path.relative_to(PRIVATE_ROOT).as_posix()
        extra_values.update(raw_numbers(raw) | raw_numbers(rel.encode('utf-8')))
        if path.suffix.lower() == '.json':
            extra_values.update(structured_numbers(decode(raw)))
        private_sources.append(dict(path=rel, sha256=sha(raw), bytes=len(raw)))
    private_sources.sort(key=lambda item: item['path'])
    protected_files.sort(key=lambda item: item['path'])
    values = (train | validation | market_holdout | market_development | protected
              | market_roles | round_roles | failed_roles | extra_values)
    return dict(values=values, market_binding_sha256=sha(market_raw),
                private_sources=private_sources, protected_files=protected_files,
                counts=dict(task_choice_train=len(train), task_choice_validation=len(validation),
                            market_formal_reserved=len(market_holdout), market_development=len(market_development),
                            protected_documents=len(protected_files), protected_seed_values=len(protected),
                            market_consumed=len(market_roles), round2_consumed=len(round_roles),
                            failed_panel_roles=len(failed_roles)))


def build_inventory():
    """Freeze recognized sources at BASE; never starts a game or selects seeds."""
    _git('merge-base', '--is-ancestor', BASE, 'HEAD')
    public, public_files, total = _public_inventory()
    local, local_files, total = _local_inventory(total)
    known = known_history()
    private_bytes = sum(item['bytes'] for item in known['private_sources'])
    total += private_bytes
    require(total <= MAX_TOTAL)
    result = dict(schema='task-pair-fresh-multi-inventory-v1', base_commit=BASE,
        public_json=public_files, artifact_json=local_files, private_sources=known['private_sources'],
        authenticated_private_documents=known['protected_files'], known_counts=known['counts'],
        market_binding_sha256=known['market_binding_sha256'], input_bytes=total,
        excluded_uint32=sorted(public | local | known['values']),
        scope='Fixed commit tracked JSON contents and paths; recognized artifacts JSON contents and paths; authenticated known private sources',
        limitation='Not a claim about unknown private non-JSON, unlisted metadata, deleted history, other machines or external experiments')
    validate_inventory(result)
    return result


def _descriptor_list(value, *, max_length=MAX_FILES):
    require(type(value) is list and len(value) <= max_length)
    names = []
    for item in value:
        require(type(item) is dict and set(item) == {'path', 'sha256', 'bytes'}
                and type(item['path']) is str and item['path']
                and '\n' not in item['path'] and '\r' not in item['path']
                and type(item['sha256']) is str and HEX.fullmatch(item['sha256'])
                and type(item['bytes']) is int and 0 < item['bytes'] <= MAX_FILE)
        path = PurePosixPath(item['path'])
        require(not path.is_absolute() and path.as_posix() == item['path']
                and all(part not in ('', '.', '..') for part in path.parts))
        names.append(item['path'])
    require(names == sorted(names) and len(set(names)) == len(names))
    return True


def validate_inventory(value):
    keys = {'schema', 'base_commit', 'public_json', 'artifact_json', 'private_sources',
            'authenticated_private_documents', 'known_counts', 'market_binding_sha256',
            'input_bytes', 'excluded_uint32', 'scope', 'limitation'}
    require(type(value) is dict and set(value) == keys
            and value['schema'] == 'task-pair-fresh-multi-inventory-v1' and value['base_commit'] == BASE
            and type(value['market_binding_sha256']) is str and HEX.fullmatch(value['market_binding_sha256'])
            and type(value['input_bytes']) is int and 0 < value['input_bytes'] <= MAX_TOTAL
            and type(value['scope']) is str and type(value['limitation']) is str)
    for field in ('public_json', 'artifact_json', 'private_sources', 'authenticated_private_documents'):
        _descriptor_list(value[field])
    require(value['public_json'] and value['artifact_json'] and len(value['private_sources']) == 4
            and len(value['authenticated_private_documents']) == 4)
    counts = value['known_counts']
    expected = {'task_choice_train': 8, 'task_choice_validation': 2, 'market_formal_reserved': 12,
                'market_development': 2, 'protected_documents': 4, 'market_consumed': 2,
                'round2_consumed': 2, 'failed_panel_roles': 2}
    require(type(counts) is dict and all(type(counts.get(k)) is int and counts[k] == v for k, v in expected.items())
            and type(counts.get('protected_seed_values')) is int and 4 <= counts['protected_seed_values'] <= 16)
    values = value['excluded_uint32']
    require(type(values) is list and values and all(type(v) is int and 0 <= v < 2**32 for v in values)
            and values == sorted(set(values)))
    require(value['input_bytes'] == sum(item['bytes'] for field in ('public_json', 'artifact_json', 'private_sources')
                                         for item in value[field]))
    return True


def source_hashes(names=REQUIRED_SOURCES):
    require(type(names) in (list, tuple) and len(names) <= MAX_SOURCES
            and all(type(name) is str for name in names) and set(REQUIRED_SOURCES).issubset(names)
            and len(set(names)) == len(names))
    result = {}
    for name in sorted(names):
        path = PurePosixPath(name)
        require(name.startswith(PREFIX) and path.as_posix() == name and len(path.parts) == 4
                and all(part not in ('', '.', '..') for part in path.parts)
                and path.suffix == '.py' and not any(char in name for char in ('\\', ':', '\n', '\r')))
        result[name] = sha(_read(REPO, REPO.joinpath(*path.parts)))
    return result


def candidate_at(key, index):
    """Versioned uint32 stream, independent of match outcomes."""
    require(type(key) is bytes and len(key) == 32 and type(index) is int and 0 <= index < MAX_DRAWS)
    digest = hmac.new(key, DOMAIN + index.to_bytes(8, 'big'), hashlib.sha256).digest()
    return int.from_bytes(digest[:4], 'big')


def select_roles(key, excluded):
    require(type(key) is bytes and len(key) == 32 and type(excluded) in (set, frozenset)
            and all(type(value) is int and 0 <= value < 2**32 for value in excluded))
    selected = []
    for index in range(MAX_DRAWS):
        number = candidate_at(key, index)
        require(type(number) is int and 0 <= number < 2**32)
        if number in excluded or number in selected:
            continue
        selected.append(number)
        if len(selected) == len(ROLES):
            return dict(zip(ROLES, selected)), index + 1
    raise ValueError('Fresh seed drawing limit reached')


def validate_binding_documents(binding, inventory, inventory_sha256):
    """Check the frozen private binding and replay its deterministic draw."""
    validate_inventory(inventory)
    keys = {'schema', 'base_commit', 'salt', 'generator_key', 'generator', 'draw_count',
            'seed_roles', 'cohorts', 'inventory_sha256', 'market_binding_sha256', 'public_source_sha256'}
    require(type(binding) is dict and set(binding) == keys
            and binding['schema'] == 'task-pair-fresh-multi-private-binding-v1'
            and binding['base_commit'] == BASE and binding['generator'] == 'hmac-sha256-counter-v1'
            and type(binding['salt']) is str and HEX.fullmatch(binding['salt'])
            and type(binding['generator_key']) is str and HEX.fullmatch(binding['generator_key'])
            and binding['salt'] != binding['generator_key']
            and type(inventory_sha256) is str and HEX.fullmatch(inventory_sha256)
            and binding['inventory_sha256'] == inventory_sha256
            and binding['market_binding_sha256'] == inventory['market_binding_sha256']
            and binding['cohorts'] == COHORTS)
    selected, draws = select_roles(bytes.fromhex(binding['generator_key']), set(inventory['excluded_uint32']))
    require(type(binding['draw_count']) is int and binding['draw_count'] == draws
            and type(binding['seed_roles']) is dict and set(binding['seed_roles']) == set(ROLES)
            and all(type(value) is int and 0 <= value < 2**32 for value in binding['seed_roles'].values())
            and binding['seed_roles'] == selected)
    sources = binding['public_source_sha256']
    require(type(sources) is dict and set(REQUIRED_SOURCES).issubset(sources)
            and len(sources) <= MAX_SOURCES and all(type(key) is str and type(value) is str
                                                   and HEX.fullmatch(value) for key, value in sources.items()))
    return True


def _write_new(path, raw):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_SNAPSHOT and Path(path).parent == PRIVATE)
    _inside(PRIVATE_ROOT, PRIVATE)
    with Path(path).open('xb') as stream:
        require(stream.write(raw) == len(raw))
        stream.flush()
        os.fsync(stream.fileno())
    require(_read(PRIVATE_ROOT, path, MAX_SNAPSHOT) == raw)


def prepare(sources=REQUIRED_SOURCES):
    """Commit a private inventory before drawing; never replace an old attempt."""
    root_info = PRIVATE_ROOT.lstat()
    require(stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode)
            and not (getattr(root_info, 'st_file_attributes', 0) & 0x400)
            and PRIVATE_ROOT.resolve(strict=True) == PRIVATE_ROOT)
    require(_git('rev-parse', 'HEAD').decode('ascii').strip() == BASE
            and not _git('diff', '--name-only', '-z').strip(b'\0')
            and not _git('diff', '--cached', '--name-only', '-z').strip(b'\0')
            and not os.path.lexists(PRIVATE))
    identities = source_hashes(sources)
    inventory = build_inventory()
    inventory_raw = canonical(inventory)
    require(len(inventory_raw) <= MAX_SNAPSHOT and source_hashes(sources) == identities
            and _git('rev-parse', 'HEAD').decode('ascii').strip() == BASE)
    PRIVATE.mkdir(exist_ok=False)
    _write_new(PRIVATE / 'inventory.local.json', inventory_raw)
    # Entropy is sampled only after the immutable inventory has been written.
    salt, key = secrets.token_hex(32), secrets.token_hex(32)
    require(salt != key)
    roles, draws = select_roles(bytes.fromhex(key), set(inventory['excluded_uint32']))
    binding = dict(schema='task-pair-fresh-multi-private-binding-v1', base_commit=BASE,
        salt=salt, generator_key=key, generator='hmac-sha256-counter-v1', draw_count=draws,
        seed_roles=roles, cohorts=COHORTS, inventory_sha256=sha(inventory_raw),
        market_binding_sha256=inventory['market_binding_sha256'], public_source_sha256=identities)
    validate_binding_documents(binding, inventory, sha(inventory_raw))
    raw = canonical(binding)
    _write_new(PRIVATE / 'binding.local.json', raw)
    _write_new(PRIVATE / 'binding-sha256.local.txt', (sha(raw) + '\n').encode('ascii'))
    require(source_hashes(sources) == identities)
    return dict(status='bound', roles=4, seed_clusters=4, games_started=0)


def verify_binding(expected_sha256=None):
    """Validate only the immutable snapshot, draw and source bytes, not history."""
    raw, binding = _json_file(PRIVATE_ROOT, PRIVATE / 'binding.local.json', 65536)
    reference = _read(PRIVATE_ROOT, PRIVATE / 'binding-sha256.local.txt', 256).decode('ascii').strip()
    require(reference == sha(raw) and (expected_sha256 is None or expected_sha256 == reference))
    inventory_raw, inventory = _json_file(PRIVATE_ROOT, PRIVATE / 'inventory.local.json', MAX_SNAPSHOT)
    validate_binding_documents(binding, inventory, sha(inventory_raw))
    require(source_hashes(tuple(binding['public_source_sha256'])) == binding['public_source_sha256'])
    return raw, binding, inventory


def verify_inputs(inventory):
    """Re-scan BASE and local inputs before first game, never during the panel."""
    validate_inventory(inventory)
    require(build_inventory() == inventory)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'verify-inputs'))
    parser.add_argument('--source', action='append', default=[])
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            sources = tuple(sorted(set(REQUIRED_SOURCES) | set(args.source)))
            result = prepare(sources)
        else:
            require(not args.source)
            _raw, _binding, inventory = verify_binding()
            if args.action == 'verify-inputs':
                require(verify_inputs(inventory))
            result = dict(status='verified', games_started=0, input_rescan=args.action == 'verify-inputs')
        print(json.dumps(result))
        return 0
    except BaseException as exc:
        kind = ('memory' if isinstance(exc, MemoryError) else 'timeout' if isinstance(exc, TimeoutError)
                else 'missing' if isinstance(exc, FileNotFoundError) else 'io' if isinstance(exc, OSError)
                else 'invalid_or_runtime')
        print(json.dumps(dict(status='failed', category=kind)), file=__import__('sys').stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
