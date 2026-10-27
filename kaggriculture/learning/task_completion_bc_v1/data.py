"""Pure in-memory episode join. Caller must attest decision-time trace provenance.

Only frozen-controller opportunities are represented. Status and seed are
metric metadata, never model inputs. This module performs no I/O or training.
"""
from decision.api import observe
import numpy as np
from learning.task_choice_v1.features import CANDIDATE_SCOPE, OPS, SCHEMA_VERSION, schema_hash, candidate_id


def collect_game(game, labels, turns, expected_schema, *, expected_turns=719):
    """Join episode origins to ordered turn records; return (schema, groups).

    game supplies key, side, seed. turns yields {step, records}; steps must be
    consecutive from zero, and records preserve callback order. Caller must
    independently establish that each trace and its planning state were captured
    before the corresponding selection. A trusted expected_schema is required;
    schema self-consistency alone cannot establish authenticity. Production must
    use the default 719 turns. Results are returned only after the
    complete stream and all episode origins have been checked.
    """
    if not isinstance(expected_schema, dict) or not expected_schema:
        raise ValueError('trusted expected schema required')
    if type(expected_turns) is not int or not 1 <= expected_turns <= 719:
        raise ValueError('invalid expected turn count')
    if type(game.get('key')) is not str or game.get('side') not in (0, 1) or type(game.get('side')) is not int or type(game.get('seed')) is not int or not 0 <= game['seed'] < 2**32:
        raise ValueError('invalid game identity')
    pending = {}
    for row in labels:
        if row.get('game') != game['key'] or row.get('side') != game['side'] or type(row.get('side')) is not int:
            raise ValueError('episode game or side mismatch')
        key = (row.get('first_step'), row.get('first_ordinal'))
        if any(type(v) is not int or v < 0 for v in key) or key in pending:
            raise ValueError('invalid or duplicate episode origin')
        if row.get('status') not in ('completed_observed', 'unknown'):
            raise ValueError('invalid status')
        pending[key] = row
    groups, schema, next_step = [], expected_schema, 0
    try:
        for turn in turns:
            step = turn['step']
            if type(step) is not int or step != next_step or step >= expected_turns or type(turn['records']) is not list:
                raise ValueError('invalid turn sequence')
            next_step += 1
            units = set()
            for ordinal, record in enumerate(turn['records']):
                inputs = record['inputs']
                if set(inputs) != {'schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names', 'shared_x', 'candidate_scope', 'observation', 'configuration', 'planned', 'candidates', 'inventory_item_order', 'shed_item_order'} or any(set(c) != {'id', 'unit', 'target', 'op', 'x'} for c in inputs['candidates']):
                    raise ValueError('unexpected input fields')
                obs, cfg, planned = inputs['observation'], inputs['configuration'], inputs['planned']
                view = observe(obs, cfg)
                if view.observation != obs or view.configuration != cfg or obs['step'] != step or obs['player'] != game['side']:
                    raise ValueError('illegal observation projection')
                if inputs['candidate_scope'] != CANDIDATE_SCOPE or set(planned) != {'seeds', 'needed_seeds', 'supply', 'projected_shed', 'unit_actions', 'available_units', 'interpretation'} or planned['interpretation'] != 'before assignment: reservations and requests, not official executed state':
                    raise ValueError('invalid planning projection')
                if inputs['inventory_item_order'] != [list(i) for i in obs['private']['inventories']] or inputs['shed_item_order'] != list(obs['private']['shed']):
                    raise ValueError('item order mismatch')
                ids = [c['id'] for c in inputs['candidates']]
                chosen = record['chosen']
                if ids.count(chosen) != 1:
                    raise ValueError('chosen candidate missing or duplicated')
                index = ids.index(chosen)
                picked = inputs['candidates'][index]
                if any(record[k] != picked[k] for k in ('unit', 'target', 'op')) or picked['unit'] in units:
                    raise ValueError('record identity mismatch or duplicate unit')
                units.add(picked['unit'])
                names = inputs['feature_names']
                shared_names = inputs['shared_feature_names']
                current = dict(schema_version=SCHEMA_VERSION, schema_sha256=schema_hash(names, shared_names), shared_feature_names=shared_names, feature_names=names, ops=list(OPS))
                if inputs['schema_version'] != SCHEMA_VERSION or inputs['schema_sha256'] != current['schema_sha256'] or len(names) != 98 or len(set(names)) != 98 or len(shared_names) != 275 or len(set(shared_names)) != 275 or (schema is not None and current != schema):
                    raise ValueError('schema mismatch')
                schema = current
                shared = np.asarray(inputs['shared_x'], dtype='<f4')
                local = np.asarray([c['x'] for c in inputs['candidates']], dtype='<f4')
                if shared.shape != (275,) or local.shape != (len(ids),98) or not np.isfinite(shared).all() or not np.isfinite(local).all() or not ids or len(ids) != len(set(ids)):
                    raise ValueError('invalid numeric groups')
                ordering = []
                for c in inputs['candidates']:
                    unit, target, op = c['unit'], c['target'], c['op']
                    if type(unit) is not int or not 0 <= unit < 32768 or type(target) is not list or len(target) != 2 or any(type(x) is not int or not 0 <= x < 32768 for x in target) or op not in OPS or c['id'] != candidate_id(unit,target,op):
                        raise ValueError('invalid candidate identity')
                    ordering.append((unit,target[1],target[0],op))
                if ordering != sorted(ordering):
                    raise ValueError('candidate order mismatch')
                selected = index
                ops = tuple(c['op'] for c in inputs['candidates'])
                names = inputs['feature_names']
                own = obs['farms'][game['side']]
                positions = [own['farmer'], *own['hands']]
                for n, candidate in enumerate(inputs['candidates']):
                    unit, target, op = candidate['unit'], candidate['target'], candidate['op']
                    if unit >= len(positions) or unit not in planned['available_units']:
                        raise ValueError('candidate unit unavailable')
                    checks = {'unit.index': unit, 'unit.x': positions[unit][0], 'unit.y': positions[unit][1], 'target.x': target[0], 'target.y': target[1]}
                    checks.update({'task.op.' + name: int(name == op) for name in OPS})
                    checks['task.distance'] = abs(positions[unit][0]-target[0]) + abs(positions[unit][1]-target[1])
                    if any(float(local[n, names.index(k)]) != v for k, v in checks.items()):
                        raise ValueError('candidate vector identity mismatch')
                row = pending.pop((step, ordinal), None)
                if row is not None:
                    if any(row[k] != record[k] for k in ('unit', 'target', 'op', 'chosen')):
                        raise ValueError('episode selection mismatch')
                    groups.append(dict(shared=shared, candidates=local, chosen=selected, candidate_ops=ops, status=row['status'], seed=game['seed']))
        if pending:
            raise ValueError('episode origin missing from trace')
    except (AssertionError, KeyError, TypeError, IndexError) as exc:
        raise ValueError('malformed trace or schema') from exc
    if next_step != expected_turns:
        raise ValueError('incomplete turn stream')
    return schema, groups


def training_groups(groups):
    """Select observed completions; unknown is not a failure label."""
    return [g for g in groups if g['status'] == 'completed_observed']
