import unittest
from unittest.mock import Mock
from .report import classify,run_after_probe,safe_job

FREEZE='a'*64
MARKER='b'*64


def clean():
    return dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                exit_code=0,error_kind=None,cleanup_error_kind=None,stop_reason=None)


class ReportTest(unittest.TestCase):
    def test_memory_stop_preserved_without_receipt_read(self):
        reader=Mock(side_effect=FileNotFoundError())
        result=classify({**clean(),'passed':False,'stop_reason':'available_memory_limit'},reader,FREEZE,MARKER)
        self.assertEqual(result['primary_kind'],'job_stopped')
        self.assertEqual(result['stop_reason'],'available_memory_limit')
        self.assertFalse(result['worker_allowed']); self.assertFalse(result['retry_allowed'])
        reader.assert_not_called()

    def test_missing_receipt_and_cleanup(self):
        self.assertEqual(classify(clean(),Mock(side_effect=FileNotFoundError()),FREEZE,MARKER)['primary_kind'],'probe_receipt_missing')
        reader=Mock()
        result=classify({**clean(),'job_members_zero':False},reader,FREEZE,MARKER)
        self.assertEqual(result['primary_kind'],'job_cleanup_unconfirmed'); reader.assert_not_called()

    def test_success(self):
        reader=Mock(return_value=dict(passed=True,venv_verified=True,job_membership_verified=True,
            training_inputs_read=False,retry_allowed=False,freeze_sha256=FREEZE,marker_sha256=MARKER))
        result=classify(clean(),reader,FREEZE,MARKER)
        self.assertTrue(result['passed']); self.assertTrue(result['worker_allowed'])
        self.assertFalse(result['retry_allowed']); reader.assert_called_once_with()

    def test_incomplete_or_wrong_probe_rejected(self):
        valid=dict(passed=True,venv_verified=True,job_membership_verified=True,training_inputs_read=False,
                   retry_allowed=False,freeze_sha256=FREEZE,marker_sha256=MARKER)
        for probe in (dict(passed=True),{**valid,'venv_verified':False},{**valid,'job_membership_verified':False},
                      {**valid,'training_inputs_read':True},{**valid,'retry_allowed':True},
                      {**valid,'freeze_sha256':'c'*64},{**valid,'marker_sha256':'d'*64},
                      {**valid,'unexpected_private_data':'x'}):
            with self.subTest(probe=probe):
                result=classify(clean(),Mock(return_value=probe),FREEZE,MARKER)
                self.assertEqual(result['primary_kind'],'probe_receipt_invalid')
                self.assertFalse(result['worker_allowed'])

    def test_consumer_never_starts_worker_on_failed_probe(self):
        worker=Mock()
        reader=Mock(side_effect=FileNotFoundError())
        result=run_after_probe({**clean(),'passed':False,'stop_reason':'available_memory_limit'},
                               reader,FREEZE,MARKER,worker)
        self.assertEqual(result['probe']['stop_reason'],'available_memory_limit')
        self.assertFalse(result['worker_started']); self.assertFalse(result['retry_allowed'])
        reader.assert_not_called(); worker.assert_not_called()

    def test_consumer_success_and_uncertain_worker_launch(self):
        receipt=dict(passed=True,venv_verified=True,job_membership_verified=True,training_inputs_read=False,
                     retry_allowed=False,freeze_sha256=FREEZE,marker_sha256=MARKER)
        worker=Mock(return_value=clean())
        result=run_after_probe(clean(),Mock(return_value=receipt),FREEZE,MARKER,worker)
        self.assertTrue(result['passed']); self.assertEqual(result['worker_result'],safe_job(clean()))
        self.assertFalse(result['worker_semantic_result_checked'])
        worker.assert_called_once_with()
        result=run_after_probe(clean(),Mock(return_value=receipt),FREEZE,MARKER,
                               Mock(return_value={**clean(),'job_members_zero':False}))
        self.assertFalse(result['passed']); self.assertTrue(result['worker_started'])
        result=run_after_probe(clean(),Mock(return_value=receipt),FREEZE,MARKER,Mock(side_effect=OSError()))
        self.assertFalse(result['passed']); self.assertEqual(result['worker_started'],'unknown_after_launch')
        self.assertFalse(result['retry_allowed'])

    def test_untrusted_job_values_fail_without_leaking(self):
        reader=Mock()
        for job in ({**clean(),'stop_reason':'unexpected/path'},
                    {**clean(),'peak_sampled_rss_bytes':'secret'},
                    {**clean(),'launcher_reaped':1}):
            with self.subTest(job=job):
                result=classify(job,reader,FREEZE,MARKER)
                self.assertEqual(result['primary_kind'],'job_receipt_invalid')
                self.assertIsNone(result['job']); self.assertFalse(result['worker_allowed'])
        reader.assert_not_called()
        result=classify({**clean(),'passed':False,'error_kind':'C:'+'/private/example'},reader,FREEZE,MARKER)
        self.assertEqual(result['job']['error_kind'],'OtherError')
        self.assertFalse(result['worker_allowed']); reader.assert_not_called()
        for key in ('error_kind','cleanup_error_kind','stop_reason'):
            partial=clean(); partial.pop(key)
            self.assertEqual(classify(partial,reader,FREEZE,MARKER)['primary_kind'],'job_receipt_invalid')
        reader.assert_not_called()

    def test_untrusted_exception_name_is_redacted(self):
        odd=type('C:'+'/private/example',(Exception,),{})
        result=classify(clean(),Mock(side_effect=odd()),FREEZE,MARKER)
        self.assertEqual(result['primary_kind'],'probe_receipt_unreadable')
        self.assertEqual(result['receipt_error_kind'],'OtherError')


if __name__=='__main__': unittest.main()
