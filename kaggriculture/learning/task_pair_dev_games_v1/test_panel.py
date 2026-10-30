"""Pure synthetic contracts: no checkpoints, games, GPU or temporary files."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_pair_dev_games_v1 import panel


def synthetic_inputs():
    return dict(
        seeds={'S02': 11017, 'S03': 22029},
        models={name: {'path': 'artifacts/synthetic/' + name + '.npz',
                       'sha256': char * 64}
                for name, char in zip(panel.POLICIES, 'abc')},
        crop={'path': 'artifacts/synthetic/crop.npz', 'sha256': 'd' * 64},
        rivals={'market': 'e' * 64, 'round2': 'f' * 64},
        worker_sha='1' * 64,
    )


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_inputs()
        self.panel = panel.Panel.from_private_contract(**self.data)

    def test_exact_24_alias_plans_and_swapped_seats(self):
        plans = self.panel.plans()
        self.assertEqual(len(plans), 24)
        self.assertEqual(len({p['game'] for p in plans}), 24)
        self.assertEqual({(p['seed_alias'], p['opponent'], p['side'], p['policy'])
                          for p in plans},
                         {(a, o, s, m) for a in panel.ALIASES for o in panel.OPPONENTS
                          for s in panel.SIDES for m in panel.POLICIES})
        self.assertTrue(all(p['split'] == 'dev' and p['parity_kind'] is None for p in plans))
        self.assertTrue(all(str(seed) not in p['game'] for p in plans for seed in self.panel.seeds))

    def test_job_checks_frozen_role_and_opponent(self):
        value = dict(self.panel.plans()[0], bot_sha256='9' * 64,
                     task_checkpoint_sha256='a' * 64, rival_sha256='e' * 64)
        self.assertEqual(self.panel.validate_plan(value, job=True), value)
        for key, wrong in (('task_checkpoint_sha256', 'b' * 64),
                           ('rival_sha256', 'f' * 64), ('bot_sha256', 'bad'),
                           ('seed_alias', 'S03'), ('seed', True), ('side', False),
                           ('split', 'holdout'), ('parity_kind', 'student')):
            candidate = dict(value, **{key: wrong})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.panel.validate_plan(candidate, job=True)
        with self.assertRaises(ValueError):
            self.panel.validate_plan(dict(value, unregistered=True), job=True)

    def test_seed_and_model_role_errors_fail_closed(self):
        for key, value in (
            ('seeds', {'S02': True, 'S03': 2}),
            ('seeds', {'S02': 2, 'S03': 2}),
            ('seeds', {'S02': -1, 'S03': 2}),
            ('seeds', {'S02': 2, 'S03': 2**32}),
            ('rivals', {'market': 'e' * 64}),
            ('worker_sha', '1' * 63),
        ):
            changed = copy.deepcopy(self.data)
            changed[key] = value
            with self.subTest(key=key, value_type=type(value)), self.assertRaises(ValueError):
                panel.Panel.from_private_contract(**changed)
        changed = copy.deepcopy(self.data)
        changed['models']['control'] = dict(changed['models']['baseline'])
        with self.assertRaises(ValueError):
            panel.Panel.from_private_contract(**changed)

    def test_paths_cannot_escape_artifacts_or_use_noncanonical_forms(self):
        bad = ('/absolute.npz', 'artifacts/../x.npz', 'artifacts//x.npz',
               'artifacts/x\\y.npz', 'artifacts/x.txt', 'C' + ':' + '/x.npz', 'source/x.npz')
        for value in bad:
            changed = copy.deepcopy(self.data)
            changed['crop']['path'] = value
            with self.subTest(path_kind=bad.index(value)), self.assertRaises(ValueError):
                panel.Panel.from_private_contract(**changed)

    def test_snapshots_input_and_keeps_private_values_out_of_repr(self):
        self.data['seeds']['S02'] = 1
        self.data['models']['baseline']['path'] = 'artifacts/other.npz'
        self.data['crop']['sha256'] = 'a' * 64
        self.assertEqual(self.panel.plans()[0]['seed'], 11017)
        self.assertEqual(self.panel.task_model(self.panel.plans()[0])[0],
                         'artifacts/synthetic/baseline.npz')
        self.assertEqual(self.panel.worker_config()['crop_checkpoint_sha256'], 'd' * 64)
        self.assertNotIn('11017', repr(self.panel))
        self.assertNotIn('synthetic', repr(self.panel))

    def test_one_crop_for_every_arm_and_no_holdout_branch(self):
        cfg = self.panel.worker_config()
        self.assertNotIn('test_seeds', cfg)
        self.assertEqual(cfg['game_limits'], panel.GAME_LIMITS)
        self.assertEqual(cfg['storage'], panel.STORAGE)
        for plan in self.panel.plans():
            self.assertEqual(self.panel.worker_config()['crop_checkpoint'], cfg['crop_checkpoint'])
            self.assertEqual(self.panel.worker_config()['crop_checkpoint_sha256'], cfg['crop_checkpoint_sha256'])
            model_path, model_sha = self.panel.task_model(plan)
            self.assertTrue(model_path.endswith('/' + plan['policy'] + '.npz'))
            self.assertEqual(len(model_sha), 64)
        cfg['game_limits']['task_forwards'] = 1
        self.assertEqual(self.panel.worker_config()['game_limits']['task_forwards'], 4000)

    def test_direct_construction_cannot_skip_contract_validation(self):
        with self.assertRaises(ValueError):
            panel.Panel((True, 2), self.panel.models, self.panel.crop,
                        self.panel.rivals, self.panel.worker_sha)


if __name__ == '__main__':
    unittest.main()
