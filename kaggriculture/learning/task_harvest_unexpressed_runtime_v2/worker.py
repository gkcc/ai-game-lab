"""Per-case isolated adaptation of the SHA/Git-pinned v1 worker.

Only three local seams change: prepare imports at _factories, inject explicit
runtime dependencies at core.run_pair, and extend the reserved case receipt.
The original worker, shared modules, output guard, limits and wire stay intact.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import ModuleType

from . import runtime as preparation


REPO = Path(__file__).resolve().parents[3]
FROZEN_COMMIT = '7bdf591cbdb4070be69f7a82d43e8202dadb661c'
FROZEN_WORKER = 'kaggriculture/learning/task_harvest_unexpressed_v1/worker.py'
WORKER_SHA256 = '03ebe5477f6005221f47451bc7efc35fdb3c394fe7a83182da6c4f4e36c15ca6'
CORE_SHA256 = 'cda31e1d228a3c081d2344214cefabbde06935d8d4239438c9d5aae4a631190e'
RECEIPT_FIELD = 'runtime_preparation'
RECEIPT_SCHEMA = 'task-harvest-runtime-case-preparation-v2'
CASE_FILE, EVENT_FILE, CASE_RESERVE = 'case.local.json', 'events.jsonl.gz', 1024*1024
MAX_SOURCE = 1024*1024
require = preparation.require


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _same_metadata(left, right):
    # Receipt object key order is irrelevant; JSON value types remain binding.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _identity():
    return dict(worker_commit=FROZEN_COMMIT, worker_path=FROZEN_WORKER,
                worker_sha256=WORKER_SHA256, core_sha256=CORE_SHA256)


def _compile_worker(raw, filename):
    require(type(raw) is bytes and len(raw) <= MAX_SOURCE and _sha(raw) == WORKER_SHA256,
            'Frozen v1 worker SHA differs')
    module = ModuleType('_harvest_runtime_v2_case_worker')
    module.__file__, module.__package__ = str(filename), 'learning.task_harvest_unexpressed_v1'
    exec(compile(raw, str(filename), 'exec', dont_inherit=True), module.__dict__)
    require(_sha(Path(module.core.__file__).read_bytes()) == CORE_SHA256,
            'Frozen v1 core SHA differs')
    return module


def _frozen_worker():
    """A fresh namespace, compiled from bytes independently bound to Git/SHA."""
    result = subprocess.run(['git', '-C', str(REPO), 'show', FROZEN_COMMIT+':'+FROZEN_WORKER],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=45, check=False)
    require(result.returncode == 0 and len(result.stdout) <= MAX_SOURCE,
            'Cannot authenticate frozen v1 worker from Git')
    path = REPO/FROZEN_WORKER
    require(path.read_bytes() == result.stdout, 'Frozen v1 worker working bytes differ from Git')
    return _compile_worker(result.stdout, path)


def _new_receipt(has_target):
    return dict(schema=RECEIPT_SCHEMA, identity=_identity(),
        phase='after_source_assets_replay_checks_before_owned_writes',
        status='not_requested' if has_target else 'not_needed',
        prepare_attempts=0, prepare_returns=0, run_pair_attempts=0, run_pair_returns=0,
        prepared=None)


def _prepare_production_runtime():
    prepared = preparation.prepare_runtime()
    preparation.validate_receipt(prepared.receipt, production=True)
    return prepared


def _adapt(module, *, has_target, guard, prepare=None):
    """Bind only this namespace; no assignments to imported module globals."""
    prepare = _prepare_production_runtime if prepare is None else prepare
    receipt, prepared = _new_receipt(has_target), None
    old_factories, old_write, old_core = module._factories, module._write_case, module.core
    facade = ModuleType('_harvest_runtime_v2_case_core')
    facade.__dict__.update(vars(old_core))

    def factories(*args, **kwargs):
        nonlocal prepared
        require(has_target and receipt['prepare_attempts'] == 0,
                'Runtime preparation was repeated or unneeded')
        guard()
        receipt['prepare_attempts'] += 1
        receipt['status'] = 'preparing'
        try:
            prepared = prepare()
            require(callable(prepared.target_trace_factory), 'Runtime preparation factory is missing')
            preparation.validate_receipt(prepared.receipt)
            receipt['prepare_returns'] += 1
            receipt['prepared'] = deepcopy(prepared.receipt)
            receipt['status'] = 'prepared'
            guard()
            return old_factories(*args, **kwargs)
        except BaseException:
            receipt['status'] = 'failed'
            raise

    def run_pair(*args, **kwargs):
        require(prepared is not None and receipt['status'] == 'prepared'
                and receipt['prepare_returns'] == 1 and receipt['run_pair_attempts'] == 0,
                'Core entered before preparation or entered twice')
        require('runtime' not in kwargs and 'target_trace_factory' not in kwargs,
                'A caller tried to replace the prepared dependencies')
        receipt['run_pair_attempts'] += 1
        result = old_core.run_pair(*args, **kwargs, runtime=prepared.runtime,
                                  target_trace_factory=prepared.target_trace_factory)
        receipt['run_pair_returns'] += 1
        return result

    def write_case(dm, out, document):
        require(RECEIPT_FIELD not in document, 'Preparation receipt already exists')
        document[RECEIPT_FIELD] = deepcopy(receipt)
        return old_write(dm, out, document)

    facade.run_pair = run_pair
    module._factories, module.core, module._write_case = factories, facade, write_case
    return receipt


def _verify_preparation(document, has_target):
    value = document.get(RECEIPT_FIELD)
    expected = _new_receipt(has_target)
    require(type(value) is dict and set(value) == set(expected), 'Missing or malformed runtime receipt')
    require(value['schema'] == expected['schema'] and _same_metadata(value['identity'], _identity())
            and value['phase'] == expected['phase'], 'Runtime preparation identity differs')
    for key in ('prepare_attempts', 'prepare_returns', 'run_pair_attempts', 'run_pair_returns'):
        require(type(value[key]) is int and value[key] == int(has_target),
                'Runtime preparation/injection counts differ')
    require(value['status'] == ('prepared' if has_target else 'not_needed'),
            'Runtime preparation was not verified')
    if has_target:
        preparation.validate_receipt(value['prepared'], production=True)
    else:
        require(value['prepared'] is None, 'Unneeded runtime preparation is present')


def execute_case(modules, context, source, selection, out, guard):
    require(callable(guard))
    guard()
    module = _frozen_worker()
    _adapt(module, has_target=selection['target'] is not None, guard=guard)
    return module.execute_case(modules, context, source, selection, out, guard)


def verify_case(modules, context, source, selection, out, guard):
    """Use the frozen wire consumer first, then the preparation receipt."""
    require(callable(guard))
    guard()
    module = _frozen_worker()
    result = module.verify_case(modules, context, source, selection, out, guard)
    _verify_preparation(result['document'], selection['target'] is not None)
    guard()
    return result
