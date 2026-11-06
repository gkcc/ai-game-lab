"""Fresh frozen-core globals plus one exact official Struct comparison seam."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
from types import ModuleType

from learning.task_harvest_unexpressed_runtime_v2 import worker as prior
from learning.task_harvest_first_step_diagnostic_v1.worker import _registry
from learning.task_harvest_first_step_diagnostic_v1.capture import bounded_json, RegisteredSources
from . import compat


REPO = Path(__file__).resolve().parents[3]
CORE_PATH = 'kaggriculture/learning/task_harvest_unexpressed_v1/core.py'
CORE_SHA256 = prior.CORE_SHA256
CORE_COMMIT = prior.FROZEN_COMMIT
CASE_FILE, EVENT_FILE, CASE_RESERVE = prior.CASE_FILE, prior.EVENT_FILE, prior.CASE_RESERVE
FIELD, SCHEMA = 'struct_compatibility', 'task-harvest-struct-compatibility-v1'
require = compat.require


def _compile_core(raw, filename):
    require(type(raw) is bytes and len(raw) <= prior.MAX_SOURCE
        and hashlib.sha256(raw).hexdigest() == CORE_SHA256, 'Frozen core bytes differ')
    module = ModuleType('_struct_compat_case_core')
    module.__file__, module.__package__ = str(filename), 'learning.task_harvest_unexpressed_v1'
    exec(compile(raw, str(filename), 'exec', dont_inherit=True), module.__dict__)
    require(module.run_pair.__globals__ is vars(module) and module._run_arm.__globals__ is vars(module),
            'Core functions do not have private globals')
    return module


def _frozen_core():
    result = subprocess.run(['git', '-C', str(REPO), 'show', CORE_COMMIT+':'+CORE_PATH],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=45, check=False)
    require(result.returncode == 0 and len(result.stdout) <= prior.MAX_SOURCE)
    path = REPO/CORE_PATH
    require(path.read_bytes() == result.stdout, 'Core working bytes differ from frozen Git')
    return _compile_core(result.stdout, path)


def _identity():
    return dict(core_commit=CORE_COMMIT, core_path=CORE_PATH, core_source_sha256=CORE_SHA256,
        binding='private_compiled_core_globals.same_request',
        original_request_digest_unchanged=True, original_output_validation_unchanged=True,
        original_frame_and_inventory_order_checks_unchanged=True,
        source_bytes_unchanged=True, execution_has_explicit_comparator_seam=True)


def _local_registry(game):
    registry = _registry(game)
    entries = []
    # The runner authenticates these new files in its committed public_sources;
    # retain their exact local source hashes and relative sites, never raw errors.
    for name in ('compat.py', 'worker.py'):
        relative = 'kaggriculture/learning/task_harvest_struct_compat_v1/'+name
        path = REPO/relative
        raw = path.read_bytes()
        require(len(raw) <= prior.MAX_SOURCE)
        entries.append((relative, raw, [path]))
    extra = RegisteredSources(entries)
    registry.paths.update(extra.paths)
    registry.sites.update(extra.sites)
    return registry


def _new_receipt():
    return dict(schema=SCHEMA, identity=_identity(), status='not_prepared', binding=None,
        comparison_counts=compat.empty_stats(), first_factual_suffix=None,
        second_suffix_gate_before_actor2=False, suffix_events_saved=dict(control=0, intervene=0),
        failure_location=None, failure_location_capture_failed=False)


class FirstSuffixGate:
    """The existing sink must return before the next suffix-loop guard passes."""
    def __init__(self, receipt, comparator, core, meter, target, sink, guard):
        self.receipt, self.comparator, self.core, self.meter = receipt, comparator, core, meter
        self.target, self.sink, self.original_guard = target, sink, guard
        self.previous_counts = comparator.snapshot()
        self.post_first_step_guards = 0

    def guard(self):
        self.original_guard()
        calls = self.meter.calls
        if calls.get('official_step_returns', 0) == 1:
            if self.receipt['first_factual_suffix'] is None:
                # Frozen core has one guard after step returns, before emit.
                self.post_first_step_guards += 1
                require(self.post_first_step_guards == 1,
                        'Second suffix reached before first original sink returned')
            else:
                require(calls.get('suffix_actor_attempts', 0) == 1,
                        'First suffix gate was not reached before second actor')
                self.receipt['second_suffix_gate_before_actor2'] = True

    def emit(self, event):
        before = self.comparator.snapshot()
        self.sink(event)  # The original complete checks/write/guard must return first.
        if event.get('kind') == 'turn' and event.get('scope') == 'suffix':
            arm = event['arm']
            require(arm in ('control', 'intervene'))
            self.receipt['suffix_events_saved'][arm] += 1
            if arm == 'control' and event['step'] == self.target['step']:
                difference = compat.delta(before, self.previous_counts)
                require(self.receipt['first_factual_suffix'] is None
                    and self.meter.calls.get('official_step_attempts') == 1
                    and self.meter.calls.get('official_step_returns') == 1
                    and difference['calls'] == difference['returns'] == difference['true_returns'] == 4
                    and difference['false_returns'] == 0
                    and difference['struct_calls'] == difference['struct_true_returns'] == 2
                    and difference['struct_root_left_calls'] == 2
                    and difference['struct_root_right_calls'] == 0
                    and all(type(player['action']) is self.comparator.struct_class for player in event['frame']),
                    'First factual step did not use the exact official Struct comparison seam')
                self.receipt['first_factual_suffix'] = dict(step=event['step'], arm='control',
                    original_sink_returned=True, official_step_attempts=1, official_step_returns=1,
                    saved_event_number=self.sink.events, comparison_delta=difference,
                    accepted_action_types=['exact_official_Struct', 'exact_official_Struct'],
                    frame_physical_sha256=self.core.frame_digest(event['frame'], physical=True))
        self.previous_counts = before


def _capture_failure(receipt, registry, error):
    try:
        receipt['failure_location'] = registry.identify(error)
    except BaseException:
        receipt['failure_location'] = None
        receipt['failure_location_capture_failed'] = True


def _adapt(module, isolated_core, registry, *, prepare=None):
    """Adapt a new Worker namespace; only the compiled core comparator changes."""
    original_comparator = isolated_core.same_request
    original_request, original_digest = isolated_core._request, isolated_core.request_digest
    original_frame = isolated_core.frame_digest
    raw_run_pair, old_write = isolated_core.run_pair, module._write_case
    receipt, comparator = _new_receipt(), None
    prepare = prior._prepare_production_runtime if prepare is None else prepare
    facade = ModuleType('_struct_compat_worker_core_facade')
    facade.__dict__.update(vars(isolated_core))

    def prepare_bound():
        nonlocal comparator
        require(comparator is None and receipt['status'] == 'not_prepared')
        try:
            prepared = prepare()
            from learning.task_student_continuation_v1 import official_step
            cls, binding = compat.authenticate(official_step.official_core, original_comparator)
            comparator = compat.Comparator(cls, original_comparator)
            isolated_core.same_request = comparator
            require(isolated_core._run_arm.__globals__['same_request'] is comparator
                and isolated_core.run_pair.__globals__ is vars(isolated_core)
                and isolated_core._request is original_request
                and isolated_core.request_digest is original_digest and isolated_core.frame_digest is original_frame)
            receipt.update(status='prepared', binding=binding)
            return prepared
        except BaseException as error:
            receipt['status'] = 'failed'
            _capture_failure(receipt, registry, error)
            raise

    def run_pair(replay, trace_factory, **kwargs):
        require(comparator is not None and receipt['status'] == 'prepared')
        receipt['status'] = 'running'
        gate = FirstSuffixGate(receipt, comparator, isolated_core, kwargs['meter'],
                               kwargs['target'], kwargs['emit'], kwargs['guard'])
        try:
            result = raw_run_pair(replay, trace_factory, **dict(kwargs, guard=gate.guard, emit=gate.emit))
            require(receipt['first_factual_suffix'] is not None
                and receipt['second_suffix_gate_before_actor2'] is True
                and receipt['suffix_events_saved'] == dict(control=48, intervene=48))
            receipt['status'] = 'completed'
            return result
        except BaseException as error:
            receipt['status'] = 'failed'
            _capture_failure(receipt, registry, error)
            raise
        finally:
            receipt['comparison_counts'] = comparator.snapshot()

    def write_case(dm, out, document):
        require(FIELD not in document)
        document[FIELD] = deepcopy(receipt)
        bounded_json(document, CASE_RESERVE-1)  # The old writer adds one newline.
        return old_write(dm, out, document)

    facade.run_pair = run_pair
    module.core, module._write_case = facade, write_case
    return receipt, prepare_bound


def execute_case(modules, context, source, selection, out, guard):
    require(callable(guard) and selection['target'] is not None)
    guard()
    module, local_core = prior._frozen_worker(), _frozen_core()
    _receipt, prepare = _adapt(module, local_core, _local_registry(modules['fresh'].GAME))
    prior._adapt(module, has_target=True, guard=guard, prepare=prepare)
    return module.execute_case(modules, context, source, selection, out, guard)


def _verify_receipt(document):
    value = document.get(FIELD)
    expected = _new_receipt()
    require(type(value) is dict and set(value) == set(expected)
        and value['schema'] == SCHEMA and prior._same_metadata(value['identity'], _identity())
        and value['status'] == 'completed' and value['failure_location'] is None
        and value['failure_location_capture_failed'] is False
        and value['second_suffix_gate_before_actor2'] is True
        and prior._same_metadata(value['suffix_events_saved'], dict(control=48, intervene=48)))
    binding = value['binding']
    require(type(binding) is dict and set(binding) == set(compat.identity())
        and all(type(binding[k]) is type(v) and binding[k] == v for k, v in compat.identity().items()))
    stats = value['comparison_counts']
    compat.validate_stats(stats)
    start, core = document['target']['step'], document['core']
    require(stats['calls'] == stats['returns'] == stats['true_returns'] == 4*start+291
        and stats['false_returns'] == 0 and stats['struct_calls'] == stats['struct_true_returns'] == 192
        and stats['struct_root_left_calls'] == 192 and stats['struct_root_right_calls'] == 0)
    first = value['first_factual_suffix']
    require(type(first) is dict and set(first) == {'step', 'arm', 'original_sink_returned',
        'official_step_attempts', 'official_step_returns', 'saved_event_number', 'comparison_delta',
        'accepted_action_types', 'frame_physical_sha256'}
        and type(first['step']) is int and first['step'] == start and first['arm'] == 'control'
        and first['original_sink_returned'] is True
        and all(type(first[k]) is int and first[k] == 1 for k in ('official_step_attempts', 'official_step_returns'))
        and type(first['saved_event_number']) is int and first['saved_event_number'] == start+2
        and first['accepted_action_types'] == ['exact_official_Struct', 'exact_official_Struct']
        and first['frame_physical_sha256'] == core['control']['physical_state_sha256'][1])
    delta = first['comparison_delta']
    compat.validate_stats(delta)
    require(delta['calls'] == delta['returns'] == delta['true_returns'] == 4
        and delta['false_returns'] == 0
        and delta['struct_calls'] == delta['struct_true_returns'] == delta['struct_root_left_calls'] == 2
        and delta['struct_root_right_calls'] == 0)
    return deepcopy(value)


def verify_case(modules, context, source, selection, out, guard):
    """Keep the frozen full 48-step consumer, then verify the explicit seam."""
    require(callable(guard) and selection['target'] is not None)
    guard()
    module = prior._frozen_worker()
    result = module.verify_case(modules, context, source, selection, out, guard)
    prior._verify_preparation(result['document'], True)
    _verify_receipt(result['document'])
    # Source identity only: do not prepare a runtime or reconstruct a model here.
    spec = importlib.util.find_spec('kaggle_environments')
    require(spec is not None and len(spec.submodule_search_locations) == 1)
    path = Path(next(iter(spec.submodule_search_locations)))/'utils.py'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == compat.UTILS_SHA256)
    guard()
    return result
