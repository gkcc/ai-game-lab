"""Small seed-role and frozen-consumer regressions; no real games or data."""
from copy import deepcopy
import unittest
from .experiment import draw_seeds
from .statistics import validate_roles,development_gate,holdout_statistics
from learning.task_dagger_once_v1 import statistics as previous
from learning.task_dagger_once_v1.test_statistics import panel


def configuration(old):
    cfg=deepcopy(old)
    cfg['training_source_seeds']=cfg.pop('train_seeds')
    cfg.pop('early_screen')
    cfg.update(known_screen_seeds=[401,402],known_screens_independent=False)
    return cfg


class EntryTests(unittest.TestCase):
    def test_only_exclusions_change_the_fixed_draw(self):
        first=draw_seeds([])
        rejected=first['validation_seeds'][0]
        second=draw_seeds([rejected])
        self.assertEqual(second['generation']['rejected'],[rejected])
        self.assertEqual(second['validation_seeds'][0],first['validation_seeds'][1])
        self.assertEqual(second['validation_seeds'][1],first['test_seeds'][0])
        self.assertNotIn(rejected,second['validation_seeds']+second['test_seeds'])

    def test_known_screen_overlap_and_false_novelty_rejected(self):
        _,old=panel();cfg=configuration(old);validate_roles(cfg)
        for name,value in [('known_screen_seeds',[201,402]),('known_screens_independent',True),
                           ('training_source_seeds',[True,102,103,104])]:
            changed=deepcopy(cfg);changed[name]=value
            with self.assertRaises(ValueError):validate_roles(changed)

    def test_same_complete_panels_keep_exact_frozen_economic_results(self):
        for split,old_fn,new_fn in [('dev',previous.development_gate,development_gate),
                                   ('holdout',previous.holdout_statistics,holdout_statistics)]:
            rows,old=panel(split=split)
            expected=old_fn(rows,old);actual=new_fn(rows,configuration(old))
            actual.pop('reused_screen_outcomes_in_panel');actual['schema']=expected['schema']
            self.assertEqual(actual,expected)
            rows[0]['seed']=401
            with self.assertRaises(ValueError):new_fn(rows,configuration(old))


if __name__=='__main__':unittest.main()
