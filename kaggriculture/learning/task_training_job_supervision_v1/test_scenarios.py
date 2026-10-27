from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch, mock_open
from . import scenarios as s


class ScenarioTest(unittest.TestCase):
    def test_launcher_child_is_hidden(self):
        for name in ('normal','rss_stop'):
            command = s.command_for(name,'synthetic-python')
            calls = []
            fake = types.ModuleType('subprocess')
            fake.CREATE_NO_WINDOW = 0x08000000
            def popen(args,**kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(wait=lambda:0)
            fake.Popen = popen
            with patch.dict(sys.modules,{'subprocess':fake}), self.assertRaises(SystemExit) as exit_code:
                exec(command[-1],{})
            self.assertEqual(exit_code.exception.code,0)
            self.assertEqual(calls,[dict(creationflags=fake.CREATE_NO_WINDOW)])
    def test_expected_results(self):
        result = dict(assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,error_kind=None,
            cleanup_error_kind=None,max_sampled_members=2,rss_threshold_sample_members=2,passed=True,exit_code=0,stop_reason=None)
        self.assertTrue(s.accepted('normal',result))
        self.assertTrue(s.accepted('rss_stop',{**result,'passed':False,'stop_reason':'rss_limit'}))
        for key,value in (('max_sampled_members',1),('launcher_reaped',False),('job_members_zero',False),('cleanup_error_kind','OSError')):
            self.assertFalse(s.accepted('normal',{**result,key:value}))

    def test_frozen_gate_partial_and_no_followup(self):
        exp = types.ModuleType('learning.task_training_job_supervision_v1.experiment')
        exp.OUT = Path(__file__).resolve().parent/'synthetic'
        exp.ROOT = exp.OUT.parent
        exp.ready = lambda _:dict(scenarios=s.SCENARIOS)
        exp.verify_commit = lambda _:None
        exp.sha = lambda _:'h'
        exp.read = lambda _:{}
        records = {}
        exp.write_new = lambda path,value:records.update({Path(path).name:value})
        with patch.dict(sys.modules,{'learning.task_training_job_supervision_v1.experiment':exp}), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open',mock_open()), \
             patch.object(s.json,'dump',side_effect=OSError()),patch.object(s,'supervise') as supervised:
            with self.assertRaises(RuntimeError): s.run_scenario('normal',lambda:None)
            supervised.assert_not_called()
            self.assertFalse(records['normal-result.json']['marker_complete'])
        with patch.object(s,'run_scenario',side_effect=RuntimeError()) as run:
            with self.assertRaises(RuntimeError): s.run_all()
            self.assertEqual(run.call_count,1)

    def test_rss_stop_requires_bound_normal_receipt(self):
        exp = types.ModuleType('learning.task_training_job_supervision_v1.experiment')
        exp.OUT = Path(__file__).resolve().parent/'synthetic'; exp.ROOT = exp.OUT.parent
        exp.ready = lambda _:dict(scenarios=s.SCENARIOS)
        exp.verify_commit = lambda _:None
        exp.sha = lambda _:'h'
        exp.write_new = lambda *args:None
        good = dict(assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,error_kind=None,
            cleanup_error_kind=None,max_sampled_members=2,passed=True,exit_code=0,stop_reason=None)
        marker = dict(attempt=1,retry_allowed=False,freeze_sha256='h')
        receipt = dict(scenario='normal',passed=True,freeze_sha256='h',marker_sha256='h',marker_created=True,
                       marker_complete=True,error_kind=None,retry_allowed=False,result=good)
        for bad in ({}, {**receipt,'passed':False}, {**receipt,'marker_sha256':'wrong'},
                    {**receipt,'result':{**good,'launcher_reaped':False}}):
            exp.read = lambda path: marker if Path(path).name == 'normal-started.json' else bad
            with patch.dict(sys.modules,{'learning.task_training_job_supervision_v1.experiment':exp}), \
                 patch.object(s,'supervise') as supervised:
                with self.assertRaises(ValueError): s.run_scenario('rss_stop',lambda:None)
                supervised.assert_not_called()

    def test_receipt_identity_error_is_recorded(self):
        exp = types.ModuleType('learning.task_training_job_supervision_v1.experiment')
        exp.OUT = Path(__file__).resolve().parent/'synthetic'; exp.ROOT = exp.OUT.parent
        exp.ready = lambda _:dict(scenarios=s.SCENARIOS)
        exp.verify_commit = lambda _:None
        exp.read = lambda _:{}
        hashes = iter(['h',OSError('marker hash')])
        def digest(_):
            value = next(hashes)
            if isinstance(value,Exception): raise value
            return value
        exp.sha = digest
        records = {}
        exp.write_new = lambda path,value:records.update({Path(path).name:value})
        result = dict(assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,error_kind=None,
            cleanup_error_kind=None,max_sampled_members=2,passed=True,exit_code=0,stop_reason=None)
        # Initial marker hash succeeds, receipt identity hash fails.
        with patch.dict(sys.modules,{'learning.task_training_job_supervision_v1.experiment':exp}), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open',mock_open()), \
             patch.object(s,'supervise',return_value=result):
            with self.assertRaises(RuntimeError): s.run_scenario('normal',lambda:None)
        self.assertEqual(records['normal-result.json']['identity_error_kind'],'OSError')
        self.assertFalse(records['normal-result.json']['passed'])


if __name__ == '__main__': unittest.main()
