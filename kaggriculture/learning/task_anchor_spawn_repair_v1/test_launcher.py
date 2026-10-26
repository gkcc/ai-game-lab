"""Small cumulative receipt regression; no subprocess or model is launched."""
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
from .launcher import Receipts,OutputBudgetExceeded
from learning.task_anchor_once_v1.test_storage import _scratch


class ReceiptTests(unittest.TestCase):
    def test_real_entry_records_original_error_even_if_wait_and_join_fail(self):
        from . import entry
        def fail_wait(**kwargs):raise TimeoutError('original target timeout')
        def fail_join(**kwargs):raise RuntimeError('secondary reader failure')
        child=SimpleNamespace(process=SimpleNamespace(poll=lambda:None,wait=fail_wait),
            thread=SimpleNamespace(join=fail_join,is_alive=lambda:True))
        with _scratch() as folder:
            out=folder/'output';out.mkdir();old=folder/'previous';old.mkdir()
            (out/'registration.json').write_text(json.dumps(dict(root_owned_names=[
                'registration.json','dispatch-started.json','cancel.local.json','dispatch-failed.json'])))
            (out/'freeze.json').write_text('{"synthetic":true}')
            (old/'gpu-launch.local.json').write_text(json.dumps(dict(distro='synthetic',workdir='synthetic',python='synthetic')))
            with patch.object(entry,'OUT',out),patch.object(entry,'PREVIOUS',old), \
                 patch.object(entry,'ready'),patch.object(entry,'CapturedProcess',return_value=child), \
                 patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=2**40)):
                with self.assertRaisesRegex(TimeoutError,'original target'):entry.run()
            receipt=json.loads((out/'dispatch-failed.json').read_text())
            self.assertEqual(receipt['kind'],'TimeoutError')
            self.assertIn('reader:RuntimeError',receipt['cleanup']['errors'])
            self.assertFalse(receipt['cleanup']['passed']);self.assertFalse(receipt['retry_allowed'])
            self.assertIsNone(receipt['cleanup']['linux_cleanup_evidence'])
            self.assertFalse((out/'dispatch-result.json').exists())
        self.assertFalse(folder.exists())

    def test_cleanup_errors_are_reported_without_replacing_original_failure(self):
        from . import entry
        process=SimpleNamespace(poll=lambda:None,wait=lambda **kw:(_ for _ in ()).throw(TimeoutError()))
        thread=SimpleNamespace(join=lambda **kw:(_ for _ in ()).throw(RuntimeError()),is_alive=lambda:True)
        with _scratch() as folder,patch.object(entry,'OUT',folder), \
             patch.object(entry,'save',side_effect=OSError()),patch.object(entry.time,'monotonic',return_value=40.):
            result=entry.finish_capture(SimpleNamespace(process=process,thread=thread),0.)
        self.assertEqual(result['errors'],['cancel:OSError','transport:TimeoutError','reader:RuntimeError'])
        self.assertFalse(result['passed']);self.assertFalse(result['within_total_deadline'])
        self.assertIsNone(result['linux_cleanup_evidence'])
        self.assertIsNone(result['transport_reaped']);self.assertIsNone(result['reader_exited'])
        self.assertEqual(result['post_deadline_recovery_seconds'],10.)

    def test_separate_write_scopes_do_not_reset_the_shared_allowance(self):
        with _scratch() as folder:
            out=folder/'receipts';receipts=Receipts(out,100)
            receipts.write('launch-started.json',{'value':'short'})
            first=receipts.used
            self.assertEqual(first,(out/'launch-started.json').stat().st_size)
            with self.assertRaises(OutputBudgetExceeded):
                receipts.write('launch-result.json',{'value':'x'*100})
            self.assertTrue(receipts.refused)
            self.assertGreaterEqual(receipts.used,first)
            self.assertLessEqual(receipts.used,100)
            with self.assertRaises(OutputBudgetExceeded):receipts.write('launch-result.json',{})
            self.assertEqual((out/'launch-started.json').read_text().count('short'),1)
        self.assertFalse(folder.exists())


if __name__=='__main__':unittest.main()
