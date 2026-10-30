"""The structural aliases must remain bound to the intended private roles."""
import copy
import unittest

from learning.task_pair_dev_campaign_v1.manifest import build_manifest
from . import bindings


def manifest(*, seeds, teacher=False, control='b'):
    models = {role: dict(path='artifacts/models/' + role + '.npz', sha256=digest * 64)
              for role, digest in (('baseline', 'a'), ('control', control), ('auxiliary', 'c'))}
    opponents = {'market': dict(path='artifacts/old/' + ('teacher' if teacher else 'market') + '.py',
                                sha256=bindings.TEACHER_SHA if teacher else 'd' * 64),
                 'round2': dict(path='artifacts/old/round2.py', sha256='e' * 64)}
    return build_manifest(seeds=seeds, models=models,
                          crop=dict(path='artifacts/models/crop.npz', sha256='f' * 64),
                          opponents=opponents, worker_sha256='0' * 64)


class BindingTests(unittest.TestCase):
    def test_authenticated_context_mutation_changes_identity(self):
        reg = {'source': 'fixed'}
        old = {'crop': {'sha256': 'a' * 64}}
        freeze = {'count': 1}
        private = {'prior': 'fixed'}
        identity = bindings.context_identity(reg, old, freeze, private)
        changed = copy.deepcopy(old); changed['crop']['sha256'] = 'b' * 64
        self.assertNotEqual(identity, bindings.context_identity(reg, changed, freeze, private))
        with self.assertRaises(ValueError):
            bindings.context_identity(reg, old, freeze, {'invalid': float('nan')})

    def test_old_panel_self_consistency_does_not_authorize_wrong_actual_inputs(self):
        previous = manifest(seeds={'S02': 11, 'S03': 12})
        correct = manifest(seeds={'S02': 10, 'S03': 11}, teacher=True)
        expected_teacher = 'artifacts/old/teacher.py'
        self.assertTrue(bindings.verify_mapping(correct, previous, 10, expected_teacher))
        incorrect = (manifest(seeds={'S02': 13, 'S03': 11}, teacher=True),
                     manifest(seeds={'S02': 10, 'S03': 11}, teacher=False),
                     manifest(seeds={'S02': 10, 'S03': 11}, teacher=True, control='9'))
        for bad in incorrect:
            plan = next(p for p in bad.panel.plans() if p['game'] == 'dev-S02-market-0-control')
            self.assertEqual(bad.panel.validate_plan(plan), plan)
            with self.subTest(), self.assertRaises(ValueError):
                bindings.verify_mapping(bad, previous, 10, expected_teacher)


if __name__ == '__main__':
    unittest.main()
