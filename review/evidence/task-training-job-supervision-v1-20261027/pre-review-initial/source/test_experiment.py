import unittest
from unittest.mock import patch
from . import experiment as e


class ExperimentTest(unittest.TestCase):
    def test_negative_parent_is_required(self):
        result = dict(passed=False,resource_supervision_accepted=False,resource_peak_of_worker_unknown=True)
        counter = dict(synthetic_runs=1,descendant_process_count=1,original_analysis_repeated=False)
        correction = dict(inaccessible_processes_present=True,real_analysis_repeated=False)
        e.validate_parent(result,counter,correction)
        with self.assertRaises(ValueError): e.validate_parent({**result,'passed':True},counter,correction)
        with self.assertRaises(ValueError): e.validate_parent(result,counter,{**correction,'inaccessible_processes_present':False})

    def test_deadline_and_dirty_head(self):
        for value in ('2000-01-01T00:00:00+00:00','2999-01-01T00:00:00'):
            with patch.object(e,'read',return_value={'deadline_utc':value}), self.assertRaises(TimeoutError): e.deadline_check()
        with patch.object(e,'git',side_effect=lambda *a:b'base\n' if a[0]=='rev-parse' else b' M file'), self.assertRaises(ValueError): e.clean_head('base')

    def test_environment_requires_project_interpreter_and_prefix(self):
        with patch.object(e.os.path,'samefile',side_effect=[True,True,False]), patch.object(e,'sha',return_value='digest'):
            result = e.environment()
            self.assertEqual(result['interpreter_sha256'],'digest')
            self.assertEqual(result['interpreter_relative'],'.venv/Scripts/python.exe')
        for values in ([False], [True,False], [True,True,True]):
            with self.subTest(values=values), patch.object(e.os.path,'samefile',side_effect=values), self.assertRaises(ValueError):
                e.environment()
        with patch.object(e.os.path,'samefile',side_effect=OSError()), self.assertRaises(ValueError): e.environment()


if __name__ == '__main__': unittest.main()
