"""Deployment-visible task inputs; labels and teacher scores live elsewhere.

Candidates are the frozen teacher's economic opportunities, NOT the official
legal action space. Features describe reservations before this assignment;
planned supplies and actions are estimates/requests, not executed resources.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from decision.api import observe, CROPS, ANIMALS, PRODUCTS, TILE_KEYS
from learning.bc_crop.features import encode as crop_encode

SCHEMA_VERSION = 'task-choice-v1'
OPS = ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')
KINDS = ('EMPTY', 'LOCKED', 'WEED', 'PLANT', 'COOP', 'PASTURE', 'OTHER')
NUMERIC_TILE_KEYS = tuple(k for k in TILE_KEYS if k not in ('kind', 'crop', 'animal'))
CANDIDATE_SCOPE = 'frozen teacher economic opportunities; not official legality masks'
# Structured inputs are part of the protocol too. Explicit item-order arrays
# survive sorted-key dataset serialization; dictionaries alone do not.
STRUCTURED_FIELDS = ('observation', 'configuration', 'planned', 'candidates',
                     'inventory_item_order', 'shed_item_order')


def schema_hash(names, shared_names=()):
    document = [SCHEMA_VERSION, list(OPS), list(KINDS), list(shared_names), list(names),
                list(STRUCTURED_FIELDS)]
    return hashlib.sha256(json.dumps(document, separators=(',', ':')).encode()).hexdigest()


def candidate_id(unit, target, op):
    return f'u{unit}:x{target[0]}:y{target[1]}:{op}'


def _tile_features(tile, prefix):
    d = tile if isinstance(tile, dict) else {}
    kind = 'EMPTY' if tile is None else 'LOCKED' if tile == 'LOCKED' else d.get('kind', 'OTHER')
    if kind not in KINDS:
        kind = 'OTHER'
    pairs = [(prefix + '.kind.' + k, float(kind == k)) for k in KINDS]
    pairs += [(prefix + '.crop.' + c, float(d.get('crop') == c)) for c in CROPS]
    pairs += [(prefix + '.animal.' + a, float(d.get('animal') == a)) for a in ANIMALS]
    for k in NUMERIC_TILE_KEYS:
        pairs.append((prefix + '.' + k + '.present', float(k in d)))
        pairs.append((prefix + '.' + k, float(d.get(k, 0))))
    return pairs


def encode_candidates(context):
    """Return callback-safe JSON and a private ID -> choice-tuple lookup.

    The second return value is controller plumbing only. It contains the source
    choice tuples, including teacher priorities, and MUST NOT be published as
    neural inputs or passed to task_selector. The first return is sufficient for
    a variable-candidate classification dataset. Teacher labels are added later.
    """
    view = observe(context['observation'], context['configuration'])
    obs, cfg = view.observation, view.configuration
    choices = context['choices']
    lookup, specs = {}, []
    for choice in choices:
        unit, task_index = choice[-2:]
        target, action, _priority = context['tasks'][task_index]
        op = action[0]
        if op not in OPS:
            raise ValueError('Unknown task operation in pinned teacher')
        identifier = candidate_id(unit, target, op)
        if identifier in lookup:
            raise ValueError('Duplicate semantic candidate identity')
        lookup[identifier] = choice
        specs.append((unit, target[1], target[0], op, identifier))
    specs.sort()  # Observable identity only, never priority/score/task list order.
    if not specs:
        raise ValueError('Task selector requires a nonempty opportunity set')

    # All global crop-controller features are shared at this assignment point.
    # Encode them once, then replace the exactly enumerated unit/target columns.
    first_unit, first_y, first_x, _op, _identifier = specs[0]
    base_context = dict(context, observation=obs, configuration=cfg,
                        i=first_unit, pos=(first_x, first_y))
    base, base_names = crop_encode(base_context)
    local_names = ['unit.index', 'unit.x', 'unit.y', 'target.x', 'target.y']
    local_names += ['carried_unit.' + item for item in PRODUCTS]
    shared_names = [name for name in base_names if name not in local_names]
    shared_x = [float(value) for name, value in zip(base_names, base) if name not in local_names]
    candidates, feature_names = [], None
    own = obs['farms'][obs['player']]
    for unit, y, x, op, identifier in specs:
        ux, uy = context['units'][unit]
        replacements = {'unit.index': unit, 'unit.x': ux, 'unit.y': uy,
                        'target.x': x, 'target.y': y}
        replacements.update({'carried_unit.' + item: obs['private']['inventories'][unit].get(item, 0)
                             for item in PRODUCTS})
        additional = [('task.op.' + kind, float(op == kind)) for kind in OPS]
        additional += [('task.distance', float(abs(ux-x) + abs(uy-y)))]
        additional += _tile_features(own['tiles'][y][x], 'target_tile')
        additional += _tile_features(own['tiles'][uy][ux], 'unit_tile')
        names = local_names + [name for name, _ in additional]
        if feature_names is not None and feature_names != names:
            raise ValueError('Nonuniform candidate feature schema')
        feature_names = names
        vector = np.asarray([replacements[name] for name in local_names]
                            + [v for _, v in additional], dtype=np.float32)
        if not np.isfinite(vector).all():
            raise ValueError('Nonfinite task input')
        candidates.append(dict(id=identifier, unit=unit, target=[x, y], op=op, x=vector.tolist()))

    planned = {k: deepcopy(context[k]) for k in ('seeds', 'needed_seeds', 'supply', 'projected_shed')}
    planned['unit_actions'] = deepcopy(context['actions'])
    planned['available_units'] = sorted(context['available'])
    planned['interpretation'] = 'before assignment: reservations and requests, not official executed state'
    inputs = dict(schema_version=SCHEMA_VERSION, schema_sha256=schema_hash(feature_names, shared_names),
                  feature_names=feature_names, shared_feature_names=shared_names, shared_x=shared_x,
                  candidate_scope=CANDIDATE_SCOPE,
                  observation=obs, configuration=cfg, planned=planned, candidates=candidates,
                  inventory_item_order=[list(inv) for inv in obs['private']['inventories']],
                  shed_item_order=list(obs['private']['shed']))
    return inputs, lookup
