"""Episode/call budget regressions using synthetic controllers, no checkpoints."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import policy


class FakeNetwork:
    calls = 0

    def __init__(self, *_):
        pass

    def __call__(self, inputs):
        type(self).calls += 1
        return inputs['candidates'][0]['id']


class FakeController:
    def __init__(self, selector, *_):
        self.selector = selector
        self.crop = SimpleNamespace(network=SimpleNamespace(logits=lambda *_: [1.]))

    def __call__(self, observation, configuration):
        inputs = {'candidates': [{'id': 'a'}]}
        if observation.get('inject_label'):
            inputs['teacher_label'] = 'a'
        chosen = self.selector(inputs) if self.selector else 'a'
        self.crop.network.logits([], [])
        self.last = dict(step=observation['step'], records=[dict(inputs=inputs,
                         chosen=chosen, teacher_label='a')], load_errors=0,
                         inference_errors=0, fallbacks=0)
        return {'farmer': ['WAIT'], 'hands': [], 'market': []}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        FakeNetwork.calls = 0
        for name, value in [('TaskNetwork', FakeNetwork), ('TaskAgent', FakeController),
                            ('observe', lambda o,c: SimpleNamespace(observation=o,configuration=c))]:
            patcher = patch.object(policy, name, value)
            patcher.start(); self.addCleanup(patcher.stop)

    def agent(self, **kw):
        return policy.BoundedAgent('crop', 'sha', 'task', 'sha', **kw)

    def test_new_episode_resets_owned_history_and_counters(self):
        a = self.agent()
        request = a({'step': 0, 'player': 0}, {})
        self.assertEqual(request, a({'step': 1, 'player': 0}, {}))
        self.assertEqual(a.last['model_counts']['task_forwards'], 2)
        self.assertEqual(request, a({'step': 0, 'player': 1}, {}))
        self.assertEqual(a.last['model_counts']['task_forwards'], 1)
        with self.assertRaises(ValueError):
            a({'step': 1, 'player': 0}, {})
        self.assertIsNone(a.last)

    def test_task_budget_and_training_label_fail_before_network_call(self):
        limits = dict(policy.LIMITS, task_forwards=1)
        a = self.agent(limits=limits)
        a({'step': 0, 'player': 0}, {})
        with self.assertRaises(RuntimeError):
            a({'step': 1, 'player': 0}, {})
        self.assertEqual(FakeNetwork.calls, 1)
        b = self.agent()
        with self.assertRaises(ValueError):
            b({'step': 0, 'player': 0, 'inject_label': True}, {})
        self.assertEqual(FakeNetwork.calls, 1)

    def test_teacher_has_no_task_network_but_counts_crop_and_controller(self):
        a = policy.BoundedAgent('crop', 'sha')
        a({'step': 0, 'player': 0}, {})
        self.assertEqual(FakeNetwork.calls, 0)
        self.assertEqual(a.last['model_counts']['task_rows'], 0)
        self.assertEqual(a.last['model_counts']['crop_forwards'], 1)
        self.assertEqual(a.last['model_counts']['controller_calls'], 1)


if __name__ == '__main__':
    unittest.main()
