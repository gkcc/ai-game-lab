"""An authenticated official-container copy, followed by the original comparator.

This is deliberately not a JSON coercion layer. Unknown container subclasses,
non-JSON objects, invalid keys and scalar types still reach the old strict
comparator unchanged and retain its rejection/type semantics.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import ModuleType


UTILS_SHA256 = '537b627b11784d424147ef57ebb0369b039bf83c9f891e81f10486b1f552334b'
OFFICIAL_CORE_SHA256 = '0922c4599a1b6e0d8c3dadf06ae5297f98138d859d686f00daee6f36e6d45d0e'
POLICY_SHA256 = 'd6844737c962a90f54414a8efbfe637b5fd314cdbd277ec3d13f26fdb81c0b1a'
UTILS_MODULE = 'kaggle_environments.utils'
STAT_KEYS = ('calls', 'returns', 'true_returns', 'false_returns', 'struct_calls',
             'struct_objects', 'struct_true_returns', 'struct_root_left_calls', 'struct_root_right_calls')


def require(value, message='Official Struct compatibility identity differs'):
    if not value:
        raise ValueError(message)


def identity():
    return dict(schema='task-harvest-struct-binding-v1', utils_module=UTILS_MODULE,
        utils_source_sha256=UTILS_SHA256, official_core_sha256=OFFICIAL_CORE_SHA256,
        struct_qualname='Struct', exact_class_only=True,
        official_core_structify_is_utils_structify=True, structify_global_is_exact_class=True,
        original_comparator_sha256=POLICY_SHA256, copies_only_for_comparison=True,
        scalar_coercion=False, json_roundtrip=False)


def authenticate(official_core, original_comparator):
    """Called after v2 runtime preparation, before protected writes; no game call."""
    utils = sys.modules.get(UTILS_MODULE)
    require(type(utils) is ModuleType and type(official_core) is ModuleType)
    utils_path, official_path = Path(utils.__file__).resolve(strict=True), Path(official_core.__file__).resolve(strict=True)
    require(utils_path.name == 'utils.py' and utils_path.parent == official_path.parent
        and hashlib.sha256(utils_path.read_bytes()).hexdigest() == UTILS_SHA256
        and hashlib.sha256(official_path.read_bytes()).hexdigest() == OFFICIAL_CORE_SHA256)
    cls, structify = utils.Struct, utils.structify
    require(type(cls) is type and cls.__module__ == UTILS_MODULE and cls.__qualname__ == 'Struct'
        and cls.__bases__ == (dict,) and official_core.structify is structify
        and structify.__globals__ is vars(utils) and structify.__globals__.get('Struct') is cls
        and Path(structify.__code__.co_filename).resolve(strict=True) == utils_path
        and Path(cls.__init__.__code__.co_filename).resolve(strict=True) == utils_path)
    require(original_comparator.__name__ == 'same_request'
        and hashlib.sha256(Path(original_comparator.__code__.co_filename).read_bytes()).hexdigest() == POLICY_SHA256)
    return cls, identity()


def empty_stats():
    return {key: 0 for key in STAT_KEYS}


def comparison_copy(value, struct_class):
    """Preserve key/array order and scalar identity; never mutate the input."""
    count = 0
    def copy(item):
        nonlocal count
        kind = type(item)
        if kind is dict or kind is struct_class:
            count += int(kind is struct_class)
            # dict.items bypasses no validation: exact dict or authenticated class.
            return {key: copy(child) for key, child in dict.items(item)}
        if kind is list:
            return [copy(child) for child in item]
        # The original _typed remains authoritative for all other types/keys.
        return item
    return copy(value), count


class Comparator:
    def __init__(self, struct_class, original):
        self.struct_class, self.original = struct_class, original
        self.stats = empty_stats()

    def __call__(self, left, right):
        self.stats['calls'] += 1
        copied_left, count_left = comparison_copy(left, self.struct_class)
        copied_right, count_right = comparison_copy(right, self.struct_class)
        count = count_left+count_right
        self.stats['struct_objects'] += count
        self.stats['struct_calls'] += int(count > 0)
        self.stats['struct_root_left_calls'] += int(type(left) is self.struct_class)
        self.stats['struct_root_right_calls'] += int(type(right) is self.struct_class)
        value = self.original(copied_left, copied_right)
        require(type(value) is bool)
        self.stats['returns'] += 1
        self.stats['true_returns' if value else 'false_returns'] += 1
        self.stats['struct_true_returns'] += int(count > 0 and value)
        return value

    def snapshot(self):
        return deepcopy(self.stats)


def validate_stats(value):
    require(type(value) is dict and set(value) == set(STAT_KEYS)
        and all(type(v) is int and v >= 0 for v in value.values())
        and value['returns'] <= value['calls']
        and value['true_returns']+value['false_returns'] == value['returns']
        and value['struct_true_returns'] <= min(value['struct_calls'], value['true_returns'])
        and value['struct_calls'] <= min(value['calls'], value['struct_objects'])
        and value['struct_root_left_calls'] <= value['struct_calls']
        and value['struct_root_right_calls'] <= value['struct_calls'])


def delta(current, previous):
    validate_stats(current)
    validate_stats(previous)
    result = {key: current[key]-previous[key] for key in STAT_KEYS}
    validate_stats(result)
    return result
