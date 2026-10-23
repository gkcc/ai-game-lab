"""Small regression contracts for task control, feature isolation and lifecycle.

Environment reset constructs observations only. No official transitions or games.
"""
from copy import deepcopy
import json
import unittest

from decision.verify_mechanisms import obs
from learning.bc_crop.audit import stable
from bots.teacher_fertilizer_terminal_v2 import agent as teacher
from .agent import TaskAgent
from .features import encode_candidates
from .cases import CASES, fixture, prepare_pair


class Interface(unittest.TestCase):
    def test_six_predeclared_action_paths(self):
        for name in CASES:
            with self.subTest(case=name):
                e, a, b, _, _, _ = prepare_pair(name)
                self.assertEqual(len(e.steps),e.state[0].observation.step+1)
                self.assertTrue(all(x is None for x in e.steps[:-1]))
                self.assertEqual(a, teacher(obs(e), e.configuration))
                self.assertNotEqual(a, b)

    def test_hidden_fields_and_labels_not_callback_inputs(self):
        e = fixture('joint_PLANT')
        original = deepcopy(obs(e)); dirty = deepcopy(original)
        dirty['seed'] = 999999
        dirty['teacher_label'] = 'SECRET'
        dirty['farms'][1]['private'] = {'seeds': {'MELON': 999999}}
        dirty['farms'][1]['inventories'] = [{'MELON': 999999}]
        dirty['farms'][0]['tiles'][0][0]['priority'] = 999999
        dirty['market']['inventory']['PRIVATE_ITEM'] = 999999
        cfg = deepcopy(e.configuration);cfg['seed'] = 888888
        captured = []
        def pick(x):
            captured.append(deepcopy(x))
            return x['candidates'][0]['id']
        a = TaskAgent(pick); first = a(original, e.configuration); clean_inputs = deepcopy(captured)
        captured.clear(); second = a(dirty, cfg)
        self.assertEqual(first, second)
        self.assertEqual(clean_inputs, captured)
        encoded = json.dumps(captured)
        self.assertNotIn('999999', encoded);self.assertNotIn('888888', encoded)
        self.assertNotIn('teacher_label', encoded);self.assertNotIn('priority', encoded)

    def test_callback_mutation_does_not_mutate_controller_or_records(self):
        e = fixture('joint_PLANT')
        def mutate(x):
            selected = x['candidates'][0]['id']
            x['observation']['private']['seeds'].clear()
            x['planned']['available_units'].clear()
            x['candidates'].clear()
            return selected
        good, bad = TaskAgent(lambda x:x['candidates'][0]['id']), TaskAgent(mutate)
        self.assertEqual(good(obs(e), e.configuration), bad(obs(e), e.configuration))
        self.assertEqual(stable(good.last), stable(bad.last))

    def test_invalid_selection_has_no_teacher_fallback(self):
        e=fixture('PLANT')
        for value in (None, 0, {}, 'missing-candidate'):
            a=TaskAgent(lambda x:value)
            with self.assertRaises(ValueError): a(obs(e), e.configuration)
            self.assertIsNone(a.last)

    def test_cross_episode_reuse_matches_new_instance(self):
        reusable = TaskAgent()
        for kind in ('joint_PLANT','HARVEST','DIG','joint_PLANT'):
            e=fixture(kind);fresh=TaskAgent()
            self.assertEqual(reusable(obs(e),e.configuration),fresh(obs(e),e.configuration))
            self.assertEqual(stable(reusable.last),stable(fresh.last))

    def test_candidate_order_and_input_independent_of_teacher_priorities(self):
        class Probe(TaskAgent):
            checked=0
            def select(self, context):
                safe,_=encode_candidates(context)
                altered=dict(context)
                altered['choices']=[(9999-i,*choice[1:]) for i,choice in enumerate(reversed(context['choices']))]
                altered['tasks']=[(pos,action,123456.) for pos,action,_ in context['tasks']]
                other,_=encode_candidates(altered)
                if safe != other:raise AssertionError('Teacher priority leaked into callback input')
                self.checked+=1
                return super().select(context)
        a=Probe();e=fixture('joint_PLANT');a(obs(e),e.configuration)
        self.assertGreaterEqual(a.checked,2)

    def test_no_candidate_does_not_invoke_callback(self):
        e=fixture('PLANT');o=deepcopy(obs(e));o['step']=718;o['day']=29;o['hour']=22
        o['private']['inventories']=[{}]
        for row in o['farms'][0]['tiles']:
            for tile in row:
                if isinstance(tile,dict) and tile.get('kind')=='PLANT':
                    tile['yield_units']=0;tile['watered_today']=True
        def fail(x):raise AssertionError('Should not choose from no candidates')
        a=TaskAgent(fail);a(o,e.configuration)
        self.assertEqual(a.last['records'],[])


if __name__=='__main__':unittest.main()
