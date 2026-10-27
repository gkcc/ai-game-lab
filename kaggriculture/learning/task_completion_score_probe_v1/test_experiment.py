import sys
import types
import unittest
from unittest.mock import patch
from . import experiment as e


class ExperimentTest(unittest.TestCase):
    def test_trace_identity_tamper(self):
        rows = [dict(seed=3342200684,side=0,opponent='round2',policy=p) for p in ('candidate','teacher_task')]
        identity = dict(path='traces/x',size=10,sha256='a'*64)
        registered = {arm:dict(trace=dict(identity)) for arm in ('candidate','teacher_task')}
        collector = types.ModuleType('learning.task_completion_bc_v1.collect')
        collector.current_game = lambda *a:(dict(trace=dict(identity)),{})
        obs = types.ModuleType('learning.task_action_observation_v1.analysis')
        obs.raw_path = lambda *a:types.SimpleNamespace(stat=lambda:types.SimpleNamespace(st_size=10))
        with patch.dict(sys.modules,{'learning.task_completion_bc_v1.collect':collector,'learning.task_action_observation_v1.analysis':obs}):
            self.assertEqual(e.raw_manifest(rows,registered)['candidate'],identity)
            registered['candidate']['trace']['sha256']='b'*64
            with self.assertRaises(ValueError): e.raw_manifest(rows,registered)
            with self.assertRaises(ValueError): e.raw_manifest([],registered)

    def test_deadline_and_head(self):
        for value in ('2000-01-01T00:00:00+00:00','2999-01-01T00:00:00'):
            with patch.object(e,'read',return_value={'deadline_utc':value}), self.assertRaises(TimeoutError): e.deadline_check()
        with patch.object(e,'git',side_effect=lambda *a:b'base\n' if a[0]=='rev-parse' else b' M file'), self.assertRaises(ValueError): e.clean_head('base')

    def test_limits(self):
        self.assertEqual(e.LIMITS,dict(raw_files=2,decode_passes_per_file=1,compressed_bytes=128*1024**2,
            decoded_bytes=1024**3,max_line_bytes=16*1024**2,max_candidates=8192,max_cpu_forwards=1,seconds=300))


if __name__ == '__main__': unittest.main()
