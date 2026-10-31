"""Freeze a global, outcome-blind plan from authenticated DAgger candidates.

Callers authenticate each source trace, checkpoint and candidate list before
supplying them. No game, model, official transition or dataset training runs
here. The returned plan contains private feature digests and label indices;
it must not be published, and contains no raw observation, features or seed.
"""
from collections import Counter
import hashlib
import json
from itertools import product

import numpy as np

from learning.task_choice_v1.features import SCHEMA_VERSION, schema_hash
from learning.task_student_collection_v1.dataset import (
    CANDIDATE_DIM, DIM, SHARED_DIM, _canonical, _digest, _key, _names, _numeric_vector,
)
from . import sampling


ROLES, OPPONENTS, WINDOWS = sampling.ROLES, sampling.OPPONENTS, sampling.WINDOWS
CELL_ORDER = tuple(product(ROLES, OPPONENTS, (0, 1), range(len(WINDOWS))))
ROSTER_DOMAIN = b'kaggriculture.task-student-dagger-collection-v1.roster\x00'
HISTORY_DOMAIN = b'kaggriculture.task-student-dagger-collection-v1.previous\x00'
STATUS = ('selected', 'empty', 'all_conflict', 'all_duplicate', 'conflict_and_duplicate')
BASE_FIELDS = frozenset(('role', 'opponent', 'side', 'window', 'status', 'candidate_count',
                         'candidates_sha256', 'selected_index'))
ITEM_FIELDS = ('step', 'ordinal', 'teacher_index', 'student_index', 'third_index',
               'raw_sha256', 'normalized_sha256', 'previous_sha256')


def _contract(schema, mean, scale, guard):
    if type(schema) is not dict or not callable(guard):
        raise ValueError('Missing checkpoint schema or resource guard')
    guard()
    if schema.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Checkpoint task schema differs')
    shared_names = _names(schema.get('shared_feature_names'), SHARED_DIM)
    names = _names(schema.get('feature_names'), CANDIDATE_DIM)
    if schema.get('schema_sha256') != schema_hash(names, shared_names):
        raise ValueError('Checkpoint task schema digest differs')
    mu, sigma = _numeric_vector(mean, DIM), _numeric_vector(scale, DIM)
    if not (sigma >= 1).all():
        raise ValueError('Invalid checkpoint normalization')
    return (shared_names, names, schema['schema_sha256']), mu, sigma


def _roster_sha(items):
    raw = json.dumps(items, ensure_ascii=True, sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode('ascii')
    return hashlib.sha256(ROSTER_DOMAIN + raw).hexdigest()


def _coordinate(role, opponent, side, window):
    if (role not in ROLES or opponent not in OPPONENTS or type(side) is not int
            or side not in (0, 1) or type(window) is not int or not 0 <= window < len(WINDOWS)):
        raise ValueError('Invalid public plan coordinate')
    return role, opponent, side, window


def _candidate(item, coordinate, contract, mu, sigma, guard):
    guard()
    role, opponent, side, window = coordinate
    if type(item) is not dict:
        raise ValueError('Invalid private ranked candidate')
    step, ordinal = item.get('step'), item.get('ordinal')
    if (type(step) is not int or not WINDOWS[window][0] <= step <= WINDOWS[window][1]
            or type(ordinal) is not int or ordinal < 0):
        raise ValueError('Candidate lies outside its registered window')
    doc, before = item.get('document'), item.get('previous_document')
    if (type(doc) is not dict or type(doc.get('step')) is not int or doc['step'] != step
            or type(doc.get('records')) is not list or ordinal >= len(doc['records'])
            or (step == 0 and before is not None)
            or (step > 0 and (type(before) is not dict or type(before.get('step')) is not int
                              or before['step'] != step - 1))):
        raise ValueError('Candidate trace or prior turn differs')
    record = doc['records'][ordinal]
    if type(record) is not dict or record.get('source') != 'callback' or type(record.get('inputs')) is not dict:
        raise ValueError('Candidate callback differs')
    inputs = record['inputs']
    shared_names, names, digest = contract
    if (inputs.get('schema_version') != SCHEMA_VERSION or inputs.get('schema_sha256') != digest
            or inputs.get('shared_feature_names') != list(shared_names)
            or inputs.get('feature_names') != list(names)):
        raise ValueError('Candidate input schema differs from checkpoint')
    shared = _numeric_vector(inputs.get('shared_x'), SHARED_DIM)
    candidates = inputs.get('candidates')
    if type(candidates) is not list or not 2 <= len(candidates) <= 256:
        raise ValueError('Candidate width exceeds registered bounds')
    ids, vectors = [], []
    for candidate in candidates:
        if type(candidate) is not dict or type(candidate.get('id')) is not str or not candidate['id']:
            raise ValueError('Invalid candidate identity')
        ids.append(candidate['id'])
        vectors.append(_numeric_vector(candidate.get('x'), CANDIDATE_DIM))
    teacher, student = record.get('teacher_label'), record.get('chosen')
    if (len(set(ids)) != len(ids) or type(teacher) is not str or type(student) is not str
            or teacher == student or teacher not in ids or student not in ids):
        raise ValueError('Candidate labels do not identify a disagreement')
    third = next((identifier for identifier in ids if identifier not in (teacher, student)), None)
    if item.get('third_candidate') != third:
        raise ValueError('Third candidate differs from frozen order')
    joined = np.concatenate((np.broadcast_to(shared, (len(ids), SHARED_DIM)), np.stack(vectors)), axis=1)
    raw32 = _canonical(joined, '<f4')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized32 = _canonical((raw32.astype(np.float64) - mu) / sigma, '<f4')
    raw_key, norm_key = _key(raw32), _key(normalized32)
    # Preserve object order: the downstream bounded controller checks the
    # prior counter schema and order. No previous value is exposed in plan.
    try:
        history = json.dumps(before, ensure_ascii=True, separators=(',', ':'),
                             allow_nan=False).encode('ascii')
    except (TypeError, ValueError, UnicodeError):
        raise ValueError('Invalid previous control state') from None
    meta = dict(step=step, ordinal=ordinal, teacher_index=ids.index(teacher),
                student_index=ids.index(student), third_index=None if third is None else ids.index(third),
                raw_sha256=_digest(b'raw-float32\x00', raw_key).hex(),
                normalized_sha256=_digest(b'training-float32\x00', norm_key).hex(),
                previous_sha256=hashlib.sha256(HISTORY_DOMAIN + history).hexdigest())
    rank = (sampling.coordinate_rank(role, opponent, side, step, ordinal), step, ordinal)
    guard()
    return {'rank': rank, 'meta': meta, 'raw': raw_key, 'normalized': norm_key, 'item': item}


def _ordered(candidates, coordinate, contract, mu, sigma, guard):
    if type(candidates) is not list or len(candidates) > sampling.PER_WINDOW:
        raise ValueError('Invalid number of ranked candidates')
    nodes = [_candidate(item, coordinate, contract, mu, sigma, guard) for item in candidates]
    if len({node['meta']['step'] for node in nodes}) != len(nodes):
        raise ValueError('Repeated step in a ranked window')
    nodes.sort(key=lambda node: node['rank'])
    return nodes


def create_plan(cells, *, schema, mean, scale, guard):
    """Build one frozen plan, streaming cells in ``CELL_ORDER``.

    Each cell has role, opponent, side, window and a sampler ``candidates``
    list. The caller can yield one cell at a time; no source document or
    observation is retained by the plan. Full-pool exact raw/normalized float32
    keys are joined transitively before any window is chosen. This is a local
    development plan, not a claim of independent evaluation or better value.
    """
    contract, mu, sigma = _contract(schema, mean, scale, guard)
    iterator, all_nodes, prepared = iter(cells), [], []
    for coordinate in CELL_ORDER:
        guard()
        try:
            cell = next(iterator)
        except StopIteration:
            raise ValueError('Incomplete fixed plan grid') from None
        if type(cell) is not dict or _coordinate(*(cell.get(k) for k in
                ('role', 'opponent', 'side', 'window'))) != coordinate:
            raise ValueError('Plan cells differ from fixed source order')
        nodes = _ordered(cell.get('candidates'), coordinate, contract, mu, sigma, guard)
        # Retain exact numeric byte keys but never source documents.
        indices = []
        for node in nodes:
            node.pop('item')
            indices.append(len(all_nodes))
            all_nodes.append(node)
        prepared.append((coordinate, indices))
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError('Unexpected extra plan cell')

    parents = list(range(len(all_nodes)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parents[b] = a

    by_raw, by_normalized = {}, {}
    for i, node in enumerate(all_nodes):
        guard()
        for lookup, key in ((by_raw, node['raw']), (by_normalized, node['normalized'])):
            if key in lookup:
                union(i, lookup[key])
            else:
                lookup[key] = i
    components = {}
    for i in range(len(all_nodes)):
        components.setdefault(root(i), []).append(i)
    conflicts = set()
    for component, members in components.items():
        triples = {(all_nodes[i]['meta']['teacher_index'], all_nodes[i]['meta']['student_index'],
                    all_nodes[i]['meta']['third_index']) for i in members}
        if len(triples) != 1:
            conflicts.add(component)

    entries, seen, status_counts, selected_roles = [], set(), Counter(), Counter()
    for coordinate, indices in prepared:
        guard()
        role, opponent, side, window = coordinate
        metadata = [all_nodes[i]['meta'] for i in indices]
        chosen = None
        conflict_count = duplicate_count = 0
        for position, i in enumerate(indices):
            component = root(i)
            if component in conflicts:
                conflict_count += 1
            elif component in seen:
                duplicate_count += 1
            else:
                chosen = position
                seen.add(component)
                break
        if chosen is not None:
            status = 'selected'
            selected_roles[role] += 1
        elif not indices:
            status = 'empty'
        elif conflict_count == len(indices):
            status = 'all_conflict'
        elif duplicate_count == len(indices):
            status = 'all_duplicate'
        else:
            status = 'conflict_and_duplicate'
        entry = dict(role=role, opponent=opponent, side=side, window=window, status=status,
                     candidate_count=len(indices), candidates_sha256=_roster_sha(metadata),
                     selected_index=chosen)
        if chosen is not None:
            entry.update(metadata[chosen])
        entries.append(entry)
        status_counts[status] += 1
    summary = dict(cells=len(entries), candidate_count=len(all_nodes), components=len(components),
                   conflicting_components=len(conflicts),
                   conflicting_candidates=sum(len(components[key]) for key in conflicts),
                   selected_count=status_counts['selected'],
                   selected_by_role={role: selected_roles[role] for role in ROLES},
                   status_counts={status: status_counts[status] for status in STATUS},
                   training=False, independent_evaluation=False)
    return dict(schema='student-dagger-private-plan-v1', entries=entries, summary=summary)


def verify_entry(entry, candidates, *, schema, mean, scale, guard):
    """Verify a frozen roster on a second scan; return only its fixed choice.

    The caller must reauthenticate the complete original source and call this
    on every window. ``None`` means the frozen window was not selected; the
    caller must not replace it or run a comparison. A returned candidate is
    private and must only be passed to the trusted continuation interface.
    """
    contract, mu, sigma = _contract(schema, mean, scale, guard)
    if type(entry) is not dict:
        raise ValueError('Invalid frozen plan entry')
    coordinate = _coordinate(*(entry.get(k) for k in ('role', 'opponent', 'side', 'window')))
    status = entry.get('status')
    if status not in STATUS:
        raise ValueError('Invalid frozen plan status')
    nodes = _ordered(candidates, coordinate, contract, mu, sigma, guard)
    metadata = [node['meta'] for node in nodes]
    if (type(entry.get('candidate_count')) is not int or entry['candidate_count'] != len(nodes)
            or entry.get('candidates_sha256') != _roster_sha(metadata)):
        raise ValueError('Ranked candidate roster differs from the frozen plan')
    selected = entry.get('selected_index')
    if status == 'selected':
        if (set(entry) != BASE_FIELDS | set(ITEM_FIELDS) or type(selected) is not int
                or not 0 <= selected < len(nodes)
                or any(entry.get(key) != metadata[selected][key] for key in ITEM_FIELDS)):
            raise ValueError('Selected candidate differs from the frozen plan')
        guard()
        return nodes[selected]['item']
    if (set(entry) != BASE_FIELDS or selected is not None
            or (status == 'empty') != (len(nodes) == 0)):
        raise ValueError('Unselected frozen plan entry differs')
    guard()
    return None
