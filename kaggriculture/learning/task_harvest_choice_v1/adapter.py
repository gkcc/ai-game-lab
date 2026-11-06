"""A bounded virtual HARVEST choice inside the frozen teacher task controller.

No weights, extra real tasks, economic masks or post-hoc request patching. The
original teacher label remains an audit reference, never a selector input.
"""
from copy import deepcopy

from decision.api import Atom, assess_unit, observe
from learning.task_choice_v1.agent import TaskAgent
from learning.task_choice_v1.features import candidate_id, encode_candidates


SCOPE = 'teacher set plus bounded direct MELON harvest alternatives; not complete official mask'


def _expanded(context):
    """Return original inputs plus virtual choices; do not mutate real tasks."""
    inputs, lookup = encode_candidates(context)
    view = observe(context['observation'], context['configuration'])
    additions, virtual = [], {}
    for candidate in inputs['candidates']:
        unit, target = candidate['unit'], candidate['target']
        if (candidate['op'] != 'WATER' or unit not in context['available']
                or tuple(context['units'][unit]) != tuple(target)
                or tuple(view.positions[unit]) != tuple(target)):
            continue
        x, y = target
        tile = view.own['tiles'][y][x]
        if not (isinstance(tile, dict) and tile.get('kind') == 'PLANT' and tile.get('crop') == 'MELON'):
            continue
        if assess_unit(view, unit, Atom('HARVEST'))[0] != 'available':
            continue
        identifier = candidate_id(unit, target, 'HARVEST')
        if identifier in lookup or identifier in virtual:
            continue
        choice = lookup[candidate['id']]
        task_index = choice[-1]
        position, _action, priority = context['tasks'][task_index]
        # Keep the task count, choices, reservations and original task order.
        # Encoding an appended task would contaminate shared tasks_remaining.
        shadow_tasks = list(context['tasks'])
        shadow_tasks[task_index] = (position, ['HARVEST'], priority)
        shadow, shadow_lookup = encode_candidates(dict(context, tasks=shadow_tasks))
        for key in ('schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names',
                    'shared_x', 'planned', 'inventory_item_order', 'shed_item_order'):
            if shadow[key] != inputs[key]:
                raise ValueError('Virtual HARVEST changed an original shared input')
        if shadow_lookup.get(identifier) != choice:
            raise ValueError('Virtual HARVEST does not refer to its original WATER slot')
        encoded = [c for c in shadow['candidates'] if c['id'] == identifier]
        if len(encoded) != 1:
            raise ValueError('Virtual HARVEST candidate identity is not unique')
        additions.append(deepcopy(encoded[0]))
        virtual[identifier] = dict(choice=choice, task_index=task_index, water_id=candidate['id'])
    # Original candidates are an unchanged prefix. New choices follow original
    # semantic WATER order (unit,y,x,op,id), not teacher scores or priorities.
    if additions:
        inputs['candidates'].extend(additions)
        inputs['candidate_scope'] = SCOPE
    return inputs, lookup, virtual


class HarvestChoiceAgent(TaskAgent):
    def __init__(self, task_selector=None, *, enabled=True):
        if type(enabled) is not bool:
            raise TypeError('enabled must be a boolean')
        self.enabled = enabled
        # No checkpoint option is exposed, and no network is constructed.
        super().__init__(task_selector=task_selector, crop_mode='teacher', checkpoint=None, expected_sha=None)
        if self.crop.mode != 'teacher' or self.crop.network is not None:
            raise ValueError('Harvest choice requires the frozen teacher crop selector')

    def select(self, context):
        if not self.enabled:
            return super().select(context)
        inputs, lookup, virtual = _expanded(context)
        teacher_choice = max(context['choices'])
        unit, task_index = teacher_choice[-2:]
        target, action, _priority = context['tasks'][task_index]
        teacher_label = candidate_id(unit, target, action[0])
        chosen = teacher_label if self.task_selector is None else self.task_selector(deepcopy(inputs))
        if not isinstance(chosen, str) or chosen not in lookup and chosen not in virtual:
            raise ValueError('Task selector returned an unknown candidate ID; no fallback')
        selected = next(c for c in inputs['candidates'] if c['id'] == chosen)
        replacement = virtual.get(chosen)
        if replacement is not None:
            index = replacement['task_index']
            position, original_action, priority = context['tasks'][index]
            if original_action != ['WATER'] or tuple(position) != tuple(selected['target']):
                raise ValueError('Selected virtual task no longer refers to its WATER source')
            # The teacher next pops this exact slot and claims the whole tile.
            # No task or action is added if an original candidate was selected.
            context['tasks'][index] = (position, ['HARVEST'], priority)
        self.records.append(dict(inputs=inputs, teacher_label=teacher_label, chosen=chosen,
            unit=selected['unit'], target=selected['target'], op=selected['op'],
            source='teacher' if self.task_selector is None else 'callback',
            harvest_choice_extension=dict(added_candidate_ids=list(virtual),
                selected_virtual_harvest=replacement is not None,
                source_water_id=replacement['water_id'] if replacement is not None else None)))
        return replacement['choice'] if replacement is not None else lookup[chosen]
