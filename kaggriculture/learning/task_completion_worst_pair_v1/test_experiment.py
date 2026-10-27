import sys
import types
import unittest
from unittest.mock import patch
from . import experiment as e


class ExperimentTest(unittest.TestCase):
    def test_worst_tie_break_and_invalid(self):
        rows = [dict(seed=3342200684,side=1,margin_difference=-2),dict(seed=3342200684,side=0,margin_difference=-2),
                dict(seed=9,side=0,margin_difference=1)]
        self.assertEqual(e.worst_pair(rows),(-2,3342200684,0))
        for bad in ([],[dict(seed=-1,side=0,margin_difference=0)],[dict(seed=1,side=2,margin_difference=0)],
                    [dict(seed=1,side=0,margin_difference=float('nan'))],rows+[rows[0]]):
            with self.assertRaises(ValueError): e.worst_pair(bad)

    def test_metadata_size_tamper(self):
        rows = [dict(seed=3342200684,side=0,opponent='round2',policy=p) for p in ('candidate','teacher_task')]
        ident = dict(path='games/x',size=4,sha256='a'*64)
        collector = types.ModuleType('learning.task_completion_bc_v1.collect')
        collector.current_game = lambda *a: (dict(replay=ident,trace=ident),{})
        analysis = types.ModuleType('learning.task_action_observation_v1.analysis')
        analysis.raw_path = lambda *a: types.SimpleNamespace(stat=lambda:types.SimpleNamespace(st_size=5))
        with patch.dict(sys.modules,{'learning.task_completion_bc_v1.collect':collector,'learning.task_action_observation_v1.analysis':analysis}):
            with self.assertRaises(ValueError): e.raw_manifest(rows)

    def test_deadline_and_head(self):
        for value in ('2000-01-01T00:00:00+00:00','2999-01-01T00:00:00'):
            with patch.object(e,'read',return_value={'deadline_utc':value}), self.assertRaises(TimeoutError): e.deadline_check()
        with patch.object(e,'git',side_effect=lambda *args:b'base\n' if args[0]=='rev-parse' else b' M changed'), self.assertRaises(ValueError): e.clean_head('base')


if __name__ == '__main__': unittest.main()
