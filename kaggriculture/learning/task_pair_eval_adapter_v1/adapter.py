"""Bridge authenticated ragged development views to the existing evaluator.

This module does not authenticate sources, masks, model weights, logits or their
provenance. The caller must independently bind the exact view, source order,
schema, masks, expected clusters, inference inputs and model before using it,
and keep them unchanged during evaluation. Results and all supervision stay
private. Existing development data are not independent game evaluations.
"""

from dataclasses import dataclass
import re

import numpy as np

from learning.task_pair_data_preflight_v1 import core
from learning.task_pair_training_v1.evaluation import evaluate


SOURCES = frozenset(('teacher_dev', 'conditional_old', 'conditional_new'))
MAX_GROUPS = 100_000
MAX_CANDIDATES = 2_000_000
MAX_GROUP_WIDTH = 256
MAX_BLOCK_GROUPS = 4_096
MAX_BLOCK_CANDIDATES = 262_144
MAX_EXPECTED_CLUSTERS = 128
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_MASK_KEYS = frozenset(('seen_in_train', 'label_conflict', 'qualifying_plant'))


@dataclass(frozen=True, repr=False)
class FeatureBlock:
    """One group interval; only shared and candidate rows are model inputs.

    ``offsets`` are relative to this block, and the interval is bookkeeping;
    neither is a model feature. The array views are not a security lock.
    """
    start: int
    stop: int
    shared: np.ndarray
    candidates: np.ndarray
    offsets: np.ndarray


@dataclass(frozen=True, repr=False)
class LogitBlock:
    """Finite, original-order flattened scores for a contiguous group range.

    Optional choices are local candidate indices and, if present, must equal
    first-maximum argmax. They cannot replace the scores needed for ranking.
    """
    start: int
    stop: int
    logits: np.ndarray
    choices: object = None


def _view(view, expected_schema_sha256):
    if (type(expected_schema_sha256) is not str
            or _SHA256.fullmatch(expected_schema_sha256) is None
            or type(view) is not core.RaggedView
            or view.schema_digest != expected_schema_sha256):
        raise ValueError('Unauthenticated schema or view contract')
    core._structure(view)
    count, rows = len(view.shared), len(view.candidates)
    if not 1 <= count <= MAX_GROUPS or not 1 <= rows <= MAX_CANDIDATES:
        raise ValueError('Development source exceeds adapter limits')
    offsets = view.offsets
    if (np.any(offsets[1:] <= offsets[:-1]) or np.any(offsets > rows)
            or np.any(np.diff(offsets) > MAX_GROUP_WIDTH)):
        raise ValueError('Invalid or oversized whole group')
    return count, rows


def feature_blocks(view, *, expected_schema_sha256, max_groups=256, max_candidates=65_536):
    """Yield only source feature arrays and group boundaries, without scoring.

    The schema digest must come from an independently authenticated schema.
    For each group a model may consume ``shared[i]`` and the corresponding
    ``candidates[offsets[i]:offsets[i+1]]``. Never send labels, clusters,
    selections or masks as model features. Caller-owned source authentication,
    immutability and deployment-visible feature semantics are still required.
    """
    count, _ = _view(view, expected_schema_sha256)
    if (type(max_groups) is not int or not 1 <= max_groups <= MAX_BLOCK_GROUPS
            or type(max_candidates) is not int
            or not 1 <= max_candidates <= MAX_BLOCK_CANDIDATES):
        raise ValueError('Invalid feature block limits')
    start = 0
    while start < count:
        stop = start
        begin = int(view.offsets[start])
        while stop < count and stop - start < max_groups:
            end = int(view.offsets[stop + 1])
            if end - begin > max_candidates:
                if stop == start:
                    raise ValueError('A whole group exceeds the block limit')
                break
            stop += 1
        end = int(view.offsets[stop])
        shared = view.shared[start:stop].view()
        candidates = view.candidates[begin:end].view()
        offsets = np.array(view.offsets[start:stop + 1] - begin, dtype=np.int64, copy=True)
        for array in (shared, candidates, offsets):
            array.setflags(write=False)
        yield FeatureBlock(start, stop, shared, candidates, offsets)
        start = stop


def _expected_clusters(value):
    if (type(value) not in (tuple, list) or not 1 <= len(value) <= MAX_EXPECTED_CLUSTERS
            or any(type(cluster) is not int or not 0 <= cluster <= 0xffffffff for cluster in value)
            or len(set(value)) != len(value)):
        raise ValueError('Invalid independently fixed cluster population')
    return tuple(value)


def _masks(value, count):
    if type(value) is not dict or set(value) != _MASK_KEYS:
        raise ValueError('Invalid source mask set')
    for mask in value.values():
        if not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(bool) or mask.shape != (count,):
            raise ValueError('Invalid source mask')
    # Copy the compact masks so the two metric calls share the same snapshot.
    result = {key: np.array(value[key], copy=True) for key in _MASK_KEYS}
    if np.any(result['label_conflict'] & ~result['seen_in_train']) or np.any(result['label_conflict']):
        raise ValueError('Registered zero-conflict gate is not satisfied')
    return result


def _assemble(view, blocks, count, rows):
    # Validate the full source shape and allocation bound before allocating.
    scores = np.empty(rows, dtype=np.float64)
    group = 0
    choices_supplied = None
    for block in blocks:
        if (type(block) is not LogitBlock or type(block.start) is not int
                or type(block.stop) is not int or block.start != group
                or not block.start < block.stop <= count
                or block.stop - block.start > MAX_BLOCK_GROUPS):
            raise ValueError('Missing, repeated or out-of-order group block')
        lo, hi = int(view.offsets[block.start]), int(view.offsets[block.stop])
        values = block.logits
        if (hi - lo > MAX_BLOCK_CANDIDATES or not isinstance(values, np.ndarray)
                or values.ndim != 1 or values.shape != (hi - lo,)
                or values.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or not bool(np.isfinite(values).all())):
            raise ValueError('Invalid finite whole-group logits')
        supplied = block.choices is not None
        if choices_supplied is not None and choices_supplied != supplied:
            raise ValueError('Choices must be supplied for every block or none')
        choices_supplied = supplied
        if supplied:
            choices = block.choices
            if (not isinstance(choices, np.ndarray) or choices.ndim != 1
                    or choices.shape != (block.stop - block.start,)
                    or choices.dtype.kind not in 'iu'):
                raise ValueError('Invalid group choices')
            for local, identifier in enumerate(range(block.start, block.stop)):
                first = int(view.offsets[identifier]) - lo
                last = int(view.offsets[identifier + 1]) - lo
                # np.argmax uses the same first-candidate tie rule as evaluate.
                if int(choices[local]) != int(np.argmax(values[first:last])):
                    raise ValueError('Choice differs from first-maximum logits')
        scores[lo:hi] = values
        group = block.stop
    if group != count:
        raise ValueError('Missing group logits')
    return scores


def evaluate_source(view, blocks, masks, *, source, expected_clusters, expected_schema_sha256):
    """Evaluate one authenticated development source in two separate subsets.

    ``blocks`` must cover every group once in source order. Only ``teacher_dev``,
    ``conditional_old`` or ``conditional_new`` may be supplied, one per call;
    conditional shards are never combined here. ``expected_clusters`` is the
    independently fixed complete population for that source, not the clusters
    observed in a subset. The old/new view and masks must already be bound to
    their authenticated source identity and ordering by the caller.

    Exact values are private. A missing or undersized cluster remains unknown
    under the existing evaluator. This function neither calls a model nor
    proves how supplied logits were obtained. A final choice alone cannot
    recover within-target ranking, so full candidate logits are mandatory.
    """
    if type(source) is not str or source not in SOURCES:
        raise ValueError('One separate development source is required')
    count, rows = _view(view, expected_schema_sha256)
    expected = _expected_clusters(expected_clusters)
    private = _masks(masks, count)
    scores = _assemble(view, blocks, count, rows)
    all_groups = np.ones(count, dtype=bool)
    unseen = ~private['seen_in_train']
    return {
        'source': source,
        'all': evaluate(view, scores, all_groups, private['qualifying_plant'], expected),
        'unseen_train': evaluate(view, scores, unseen, private['qualifying_plant'], expected),
    }
