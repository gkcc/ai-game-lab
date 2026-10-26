"""Small cumulative receipt regression; no subprocess or model is launched."""
import unittest
from .launcher import Receipts,OutputBudgetExceeded
from learning.task_anchor_once_v1.test_storage import _scratch


class ReceiptTests(unittest.TestCase):
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
