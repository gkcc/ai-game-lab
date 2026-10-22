"""Focused crop-controller integration regressions, using official transitions."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
from decision.verify_mechanisms import env, obs, PASS
from .agent import CropAgent, ROOT, TEACHER_SHA
from .features import CROPS


def original():
    ns = {'__name__': 'frozen_teacher'}
    exec((ROOT / 'bots/teacher_fertilizer_terminal_v2.py').read_bytes(), ns)
    return ns['agent']


class Fixed:
    sha = 'test-only-fixed-choice'
    def __init__(self, crop): self.crop = crop
    def logits(self, x, names):
        return np.asarray([1. if c == self.crop else 0. for c in CROPS])


class Integration(unittest.TestCase):
    def test_teacher_same_input_and_resources(self):
        e = env()
        f, p = e.state[0].observation.farms[0], e.state[0].observation.private
        f['hands'] = [[3, 4], [4, 3]]
        p['inventories'] += [{}, {}]
        a, teacher = CropAgent(), original()
        for money in (40, 3000):
            for seed_count in (0, 1, 3):
                f['money'] = money
                p['seeds'] = {c: seed_count for c in CROPS}
                for step in (0, 600, 695, 700, 717, 718):
                    o = deepcopy(obs(e)); o.update(step=step, day=step//24, hour=step%24)
                    self.assertEqual(a(o, e.configuration), teacher(o, e.configuration))

    def test_missing_seed_purchase_then_official_plant(self):
        for crop in CROPS:
            e = env(); a = CropAgent(); a.network = Fixed(crop)
            first = a(obs(e), e.configuration)
            self.assertEqual(a.records[0]['outcome'], 'request_seed')
            self.assertIn(['BUY_SEED', crop, 1], first['market'])
            self.assertEqual(first['farmer'], ['PASS'])
            e.step([first, PASS])
            self.assertEqual(e.state[0].observation.private.seeds[crop], 1)
            second = a(obs(e), e.configuration)
            self.assertEqual(second['farmer'], ['PLANT', crop])
            e.step([second, PASS])
            self.assertEqual(e.state[0].observation.farms[0].tiles[4][4]['crop'], crop)
            purchased = sum(a[2] for a in second['market'] if a[:2] == ['BUY_SEED', crop])
            self.assertEqual(e.state[0].observation.private.seeds[crop], purchased)

    def test_sequential_reservation_context(self):
        e = env(); f = e.state[0].observation.farms[0]; p = e.state[0].observation.private
        f['hands'] = [[3, 4], [4, 3]]; p['inventories'] += [{}, {}]
        p['seeds']['MELON'] = 1
        a = CropAgent(); a.network = Fixed('MELON')
        result = a(obs(e), e.configuration)
        self.assertEqual(len(a.records), 3)
        self.assertEqual([r['selected_seed_before'] for r in a.records], [1, 0, 0])
        self.assertEqual([r['prior']['supply']['MELON'] for r in a.records], [6, 12, 18])
        self.assertEqual([r['prior']['needed_seeds'].get('MELON', 0) for r in a.records], [0, 0, 1])
        self.assertIn(['BUY_SEED', 'MELON', 2], result['market'])
        e.step([result, PASS])
        self.assertEqual(e.state[0].observation.private.seeds['MELON'], 2)

    def test_economic_rejection_never_substitutes_teacher(self):
        e = env(); o = deepcopy(obs(e)); o.update(step=624, day=26, hour=0)
        a = CropAgent(); a.network = Fixed('MELON')
        result = a(o, e.configuration)
        self.assertGreater(len(a.records), 0)
        self.assertTrue(all(r['outcome'] == 'economic_reject' for r in a.records))
        self.assertTrue(all(r['chosen'] == 'MELON' for r in a.records))
        self.assertFalse(any(o[0] == 'BUY_SEED' for o in result['market']))
        self.assertEqual(result['farmer'], ['PASS'])

    def test_actor_privacy_and_seed_invariance(self):
        e = env(); a = CropAgent(); a.network = Fixed('CARROT')
        o, cfg = deepcopy(obs(e)), deepcopy(dict(e.configuration))
        before = a(o, cfg); records = deepcopy(a.records)
        e.state[1].observation.private.seeds['MELON'] = 98765
        e.state[1].observation.private.shed['STRAWBERRY'] = 43210
        o2 = deepcopy(obs(e)); o2['farms'][1]['private'] = {'seeds': 999999}
        cfg['seed'] = 2147483640
        after = a(o2, cfg)
        self.assertEqual(before, after)
        self.assertEqual([r['x'] for r in records], [r['x'] for r in a.records])
        self.assertEqual([r['logits'] for r in records], [r['logits'] for r in a.records])

    def test_no_episode_memory(self):
        a = CropAgent(); e = env(); o = deepcopy(obs(e))
        first = a(o, e.configuration); x = deepcopy(a.records)
        advanced = deepcopy(o); advanced.update(step=718, day=29, hour=22)
        a(advanced, e.configuration)
        self.assertEqual(first, a(o, e.configuration))
        self.assertEqual([r['x'] for r in x], [r['x'] for r in a.records])


if __name__ == '__main__':
    unittest.main()
