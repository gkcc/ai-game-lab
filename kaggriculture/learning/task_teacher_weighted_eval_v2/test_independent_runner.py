import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

exp=types.ModuleType('learning.task_teacher_weighted_eval_v2.experiment')
exp.OUT=exp.ROOT=Path(__file__).resolve().parent/'synthetic'
for name in ('ready','verify_commit','sha','read','write_new','deadline_check'): setattr(exp,name,lambda *a:None)
resources=types.ModuleType('learning.task_teacher_weighted_eval_v2.resources')
resources.memory_ready=lambda _:True
campaign=types.ModuleType('learning.task_teacher_weighted_eval_v2.campaign')
campaign.validate_panel=lambda *a:None
spec=importlib.util.spec_from_file_location('learning.task_teacher_weighted_eval_v2._independent_test',Path(__file__).with_name('independent_runner.py'))
runner=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_teacher_weighted_eval_v2.experiment':exp,
    'learning.task_teacher_weighted_eval_v2.resources':resources}): spec.loader.exec_module(runner)


class IndependentRunnerTests(unittest.TestCase):
    def test_changed_panel_stops_before_marker_and_supervisor(self):
        with patch.dict(sys.modules,{'learning.task_teacher_weighted_eval_v2.campaign':campaign}), \
             patch.object(runner,'ready',return_value={}),patch.object(runner,'verify_commit'), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'open') as open_file, \
             patch.object(campaign,'validate_panel',side_effect=ValueError('audit drift')), \
             patch.object(runner,'supervise') as supervise,patch.object(runner,'write_new') as write:
            with self.assertRaises(ValueError): runner.run(exp.OUT)
            open_file.assert_not_called(); supervise.assert_not_called(); write.assert_not_called()


if __name__=='__main__': unittest.main()
