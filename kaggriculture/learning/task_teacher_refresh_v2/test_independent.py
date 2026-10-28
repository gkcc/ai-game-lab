import importlib.util
from pathlib import Path
import sys
import types
import unittest
import numpy
from unittest.mock import patch

exp=types.ModuleType('learning.task_teacher_refresh_v2.experiment')
exp.OUT=exp.ROOT=Path(__file__).resolve().parent/'synthetic'
for name in ('read','ready','sha','verify_commit','write_new','deadline_check'): setattr(exp,name,lambda *a:None)
campaign=types.ModuleType('learning.task_teacher_refresh_v2.campaign')
campaign.memory_ready=lambda _:True
campaign.validate_panel=lambda *a:None
def module(name):
    spec=importlib.util.spec_from_file_location('learning.task_teacher_refresh_v2._'+name+'_test',Path(__file__).with_name(name+'.py'))
    result=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'learning.task_teacher_refresh_v2.experiment':exp,'learning.task_teacher_refresh_v2.campaign':campaign}): spec.loader.exec_module(result)
    return result
method=module('independent_method'); runner=module('independent_runner')


class IndependentTests(unittest.TestCase):
    def test_teacher_label_and_fixed_20_sample(self):
        method.validate_teacher_record(dict(source='teacher',teacher_label='x',chosen='x'))
        for record in (dict(source='callback',teacher_label='x',chosen='x'),dict(source='teacher',teacher_label='x',chosen='y')):
            with self.assertRaises(ValueError): method.validate_teacher_record(record)
        identities=[('g',i,0) for i in range(40)]
        self.assertEqual(method.sample_identities(identities),method.sample_identities(identities[::-1]))
        self.assertEqual(len(method.sample_identities(identities)),20)
        with self.assertRaises(ValueError): method.sample_identities(identities+[identities[0]])

    def test_panel_drift_prevents_marker_and_supervision(self):
        with patch.dict(sys.modules,{'learning.task_teacher_refresh_v2.campaign':campaign}), \
             patch.object(runner,'ready',return_value={}),patch.object(runner,'verify_commit'), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open') as opened, \
             patch.object(campaign,'validate_panel',side_effect=ValueError('drift')),patch.object(runner,'supervise') as supervise:
            with self.assertRaises(ValueError): runner.run(exp.OUT)
            opened.assert_not_called(); supervise.assert_not_called()

    def test_low_memory_before_marker_never_starts_job(self):
        with patch.dict(sys.modules,{'learning.task_teacher_refresh_v2.campaign':campaign}), \
             patch.object(runner,'ready',return_value={}),patch.object(runner,'verify_commit'), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open') as opened, \
             patch.object(campaign,'validate_panel'),patch.object(runner,'memory_ready',return_value=False), \
             patch.object(runner,'supervise') as supervise:
            with self.assertRaisesRegex(RuntimeError,'preflight memory'): runner.run(exp.OUT)
            opened.assert_not_called(); supervise.assert_not_called()

    def test_verification_requires_teacher_target_and_manifest(self):
        cfg=dict(train_seeds=[1,2],crop_checkpoint_sha256='crop')
        result=dict(passed=True,game='dev-1-round2-0-teacher_task',checkpoint_sha256=None,crop_checkpoint_sha256='crop',
            freeze_sha256='hash',registration_sha256='hash',games_result_sha256='hash',game_ledger_sha256='hash',
            audit_ledger_sha256='hash',method_sha256='hash',source_manifest_sha256='hash',official_transitions=719,
            sample_count=20,matched=20,raw_reads=dict(files_opened=2,files_completed=2,raw_decode_passes_per_file=1))
        with patch.object(runner,'sha',return_value='hash'):
            runner.validate_verification(exp.OUT,cfg,result)
            for change in (dict(game='other'),dict(checkpoint_sha256='task'),dict(source_manifest_sha256='changed'),dict(matched=19)):
                with self.assertRaises(ValueError): runner.validate_verification(exp.OUT,cfg,{**result,**change})


if __name__=='__main__': unittest.main()
