"""Prepare real imports before the existing owned-write boundary, without a game."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


SCHEMA = 'task-harvest-runtime-prepared-v2'
ENGINE_SHA256 = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
ENGINE_CORE_SHA256 = '0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e'
EXECUTION_PUBLIC_SHA256 = 'e1b8acb32ef0b31c0de3048d481d84babf3b07510d18a9d3b2fd3da471d8f239'
EXECUTION_RAW_SHA256 = {
    'game': 'cbf38b17ca9c4c1c4f82ddb583ff59e4f605a0e8d8e971ef5bfc3664b2c76378',
    'public': EXECUTION_PUBLIC_SHA256,
}
MAPPING_PATH = 'review/evidence/task-teacher-merged-eval-v3-formal-20261030/freeze.json'
MAPPING_SHA256 = '249d72b0770bb97cb4a1cc66fc14586dd674c778185f6aebf8c50c73fad9cb4d'
REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent/'kaggriculture'


def require(condition, message='Runtime preparation differs'):
    if not condition:
        raise ValueError(message)


def expected_receipt(origin='game'):
    """Identity and scope, not an assertion that a game/model has executed."""
    require(origin in EXECUTION_RAW_SHA256, 'Unknown execution source origin')
    return dict(schema=SCHEMA, engine_sha256=ENGINE_SHA256,
        engine_core_sha256=ENGINE_CORE_SHA256,
        execution_trace_source=origin, execution_trace_raw_sha256=EXECUTION_RAW_SHA256[origin],
        execution_trace_public_sha256=EXECUTION_PUBLIC_SHA256,
        execution_trace_line_endings='CRLF' if origin == 'game' else 'LF',
        execution_trace_mapping_path=MAPPING_PATH, execution_trace_mapping_sha256=MAPPING_SHA256,
        runtime_interpreter_matches_rules=True, target_factory_lazy_imports=False,
        preparation_scope='runtime imports, identity checks and bound trace factory only',
        preparation_model_constructions=0, preparation_environment_constructions=0,
        preparation_official_transitions=0)


def validate_receipt(value, *, production=False):
    require(type(value) is dict and value.get('execution_trace_source') in EXECUTION_RAW_SHA256,
            'Unknown prepared execution source')
    origin = value['execution_trace_source']
    require(not production or origin == 'game', 'Public-only binding is not a production game runtime')
    require(json.dumps(value, sort_keys=True, allow_nan=False)
            == json.dumps(expected_receipt(origin), sort_keys=True, allow_nan=False),
            'Prepared source receipt differs')


def _execution_bytes(raw, public_raw, origin):
    """Both raw identities are pinned; normalization never replaces that check."""
    require(origin in EXECUTION_RAW_SHA256 and type(raw) is bytes and type(public_raw) is bytes)
    require(hashlib.sha256(raw).hexdigest() == EXECUTION_RAW_SHA256[origin],
            'Execution raw source identity differs')
    require(hashlib.sha256(public_raw).hexdigest() == EXECUTION_PUBLIC_SHA256
            and raw.replace(b'\r\n', b'\n') == public_raw,
            'Execution raw-to-public mapping differs')


def _execution_origin(path):
    current = Path(path).resolve(strict=True)
    locations = {'game': GAME/'decision/execution.py',
                 'public': REPO/'kaggriculture/decision/execution.py'}
    matches = [origin for origin, expected in locations.items()
               if current == expected.resolve(strict=True)]
    require(len(matches) == 1, 'Execution module has an unexpected source path')
    return matches[0]


@dataclass(frozen=True)
class Prepared:
    runtime: object
    target_trace_factory: object
    receipt: dict


def prepare_runtime():
    """Use production initialization; never construct an actor or environment.

The old initializer may import libraries that open os.devnull for output.
Those imports belong before owned_writes, whose refusal remains unchanged.
The returned factory closes over loaded objects and has no delayed import.
"""
    from learning.task_anchor_once_v1 import storage
    require(storage._ACTIVE is None, 'Runtime preparation must precede owned_writes')
    from learning.task_harvest_unexpressed_v1 import core
    runtime = core.official_runtime()
    from decision import execution
    from learning.task_student_continuation_v1 import official_step
    rules, official_core, tracer = official_step.rules, official_step.official_core, execution.ExecutionTrace
    for module, expected in ((rules, ENGINE_SHA256), (official_core, ENGINE_CORE_SHA256)):
        require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected,
                'Prepared official runtime or execution source changed')
    mapping_raw = (REPO/MAPPING_PATH).read_bytes()
    require(hashlib.sha256(mapping_raw).hexdigest() == MAPPING_SHA256, 'Historical execution mapping changed')
    mapping = json.loads(mapping_raw)
    require(mapping['runtime_files']['decision/execution.py'] == EXECUTION_RAW_SHA256['game']
            and mapping['public_runtime_sha256']['decision/execution.py'] == EXECUTION_PUBLIC_SHA256,
            'Historical execution source identities differ')
    origin = _execution_origin(execution.__file__)
    _execution_bytes(Path(execution.__file__).read_bytes(),
                     (REPO/'kaggriculture/decision/execution.py').read_bytes(), origin)
    require(runtime.interpreter is rules.interpreter)

    def target_trace_factory(environment):
        require(environment.interpreter is runtime.interpreter,
                'Target tracer and prepared runtime have different interpreters')
        return tracer(rules).installed(environment)

    return Prepared(runtime, target_trace_factory, deepcopy(expected_receipt(origin)))
