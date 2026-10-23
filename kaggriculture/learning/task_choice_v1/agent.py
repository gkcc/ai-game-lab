"""Pinned teacher with independent, replaceable task and crop selectors.

task_selector(inputs) returns an existing string candidate ID. It never sees the
teacher label and invalid output raises: there is no hidden teacher fallback.
Task opportunities, forced transport/pickup, navigation and economics retain the
frozen controller's limitations. They are not a complete official action mask.
"""
import ast
from copy import deepcopy
import hashlib

from decision.api import observe
from learning.bc_crop.agent import CropAgent, ROOT, TEACHER_SHA
from .features import candidate_id, encode_candidates, SCHEMA_VERSION


class TaskAgent:
    def __init__(self, task_selector=None, crop_mode='teacher', checkpoint=None, expected_sha=None):
        if task_selector is not None and not callable(task_selector):
            raise TypeError('task_selector must be callable or None')
        self.task_selector = task_selector
        self.crop = CropAgent(mode=crop_mode, checkpoint=checkpoint, expected_sha=expected_sha)
        self.records, self.last = [], None
        raw = (ROOT / 'bots/teacher_fertilizer_terminal_v2.py').read_bytes()
        if hashlib.sha256(raw).hexdigest() != TEACHER_SHA:
            raise ValueError('Frozen teacher changed')
        tree = ast.parse(raw)

        class Inject(ast.NodeTransformer):
            tasks = crops = returns = 0

            def visit_Assign(patch, node):
                value = node.value
                if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                        and value.func.id == 'max'):
                    if (len(value.args) == 1 and isinstance(value.args[0], ast.Name)
                            and value.args[0].id == 'choices'):
                        patch.tasks += 1
                        node.value = ast.parse('_task_select(locals())', mode='eval').body
                    elif (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                          and node.targets[0].id == 'c'):
                        patch.crops += 1
                        node.value = ast.parse('_crop_select(locals())', mode='eval').body
                return patch.generic_visit(node)

            def visit_Return(patch, node):
                if (isinstance(node.value, ast.Dict)
                        and all(isinstance(k, ast.Constant) for k in node.value.keys)
                        and [k.value for k in node.value.keys] == ['farmer', 'hands', 'market']):
                    patch.returns += 1
                    node.value = ast.Call(ast.Name('_task_finish', ast.Load()),
                                          [ast.Call(ast.Name('locals', ast.Load()), [], []), node.value], [])
                return patch.generic_visit(node)

        patch = Inject()
        tree = patch.visit(tree)
        if (patch.tasks, patch.crops, patch.returns) != (1, 1, 1):
            raise ValueError('Unexpected frozen controller integration points')
        namespace = {'__name__': 'task_choice_frozen_controller', '_task_select': self.select,
                     '_crop_select': self.crop.select, '_task_finish': self.finish}
        exec(compile(ast.fix_missing_locations(tree), '<hash-pinned-task-controller>', 'exec'), namespace)
        self.controller = namespace['agent']

    def select(self, context):
        inputs, lookup = encode_candidates(context)
        teacher_choice = max(context['choices'])
        unit, task_index = teacher_choice[-2:]
        target, action, _priority = context['tasks'][task_index]
        teacher_label = candidate_id(unit, target, action[0])
        chosen = (teacher_label if self.task_selector is None
                  else self.task_selector(deepcopy(inputs)))
        if not isinstance(chosen, str) or chosen not in lookup:
            raise ValueError('Task selector returned an unknown candidate ID; no fallback')
        selected = next(c for c in inputs['candidates'] if c['id'] == chosen)
        self.records.append(dict(inputs=inputs, teacher_label=teacher_label, chosen=chosen,
                                 unit=selected['unit'], target=selected['target'], op=selected['op'],
                                 source='teacher' if self.task_selector is None else 'callback'))
        return lookup[chosen]

    def finish(self, context, result):
        self.crop.finish(context, result)
        for record in self.records:
            record['unit_request'] = deepcopy(context['actions'][record['unit']])
        self.last = dict(schema_version=SCHEMA_VERSION, step=context['observation']['step'],
                         records=self.records, crop_records=deepcopy(self.crop.last['records']),
                         final_reservations=deepcopy(self.crop.last['final_reservations']),
                         checkpoint_sha256=self.crop.last['checkpoint_sha256'],
                         load_errors=0, inference_errors=0, fallbacks=0)
        self.last['final_reservations']['projected_shed'] = deepcopy(context['projected_shed'])
        self.last['final_reservations']['unit_actions'] = deepcopy(context['actions'])
        return result

    def __call__(self, observation, configuration):
        # Stateless owned buffers even when seat, turn, or whole episode changes.
        # A caller-provided callable owns its own state; this adapter never retains
        # prior observations and never presents labels to that callable.
        self.records, self.last = [], None
        self.crop.records, self.crop.last = [], None
        view = observe(observation, configuration)
        return self.controller(view.observation, view.configuration)
