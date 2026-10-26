"""Actual tiny child-output paths and frozen-entry logging adaptation."""
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

from .processes import CapturedProcess
from .storage import owned_writes
from .test_storage import _scratch
from .games import ROOT,official_entry


class ProcessTests(unittest.TestCase):
    def test_native_output_is_captured_and_joined_within_writer_guard(self):
        with _scratch() as folder:
            out=folder/'output'
            with owned_writes(out,4096) as budget:
                child=CapturedProcess([sys.executable,'-B','-c','import os;os.write(1,b"abc");os.write(2,b"def")'],
                    cwd=folder,log=out/'normal.log',limit_bytes=1024)
                try:
                    child.process.wait(timeout=10);receipt=child.finish()
                finally:child.stop()
                self.assertEqual((out/'normal.log').read_bytes(),b'abcdef')
                self.assertEqual(receipt['written_bytes'],6)
                self.assertEqual(budget.used_bytes,6)
        self.assertFalse(folder.exists())

    def test_overflow_is_a_failure_without_unbounded_log_or_reader(self):
        with _scratch() as folder:
            out=folder/'output'
            with owned_writes(out,4096):
                child=CapturedProcess([sys.executable,'-B','-c','import os;os.write(1,b"x"*5000)'],
                    cwd=folder,log=out/'overflow.log',limit_bytes=100)
                try:
                    child.process.wait(timeout=10)
                    with self.assertRaises(RuntimeError):child.finish()
                finally:child.stop()
                self.assertEqual((out/'overflow.log').stat().st_size,100)
                self.assertEqual(child.seen,5000)
                self.assertFalse(child.thread.is_alive())
        self.assertFalse(folder.exists())

    def test_only_native_log_redirection_is_adapted_and_restored(self):
        sentinel=object()
        fake=types.SimpleNamespace(__file__=str(ROOT/'run_match.py'),native_output=sentinel)
        def main():
            with fake.native_output(None):pass
            raise ValueError('synthetic main failure')
        fake.main=main
        with patch.dict(sys.modules,run_match=fake):
            with self.assertRaisesRegex(ValueError,'synthetic'):official_entry()
        self.assertIs(fake.native_output,sentinel)


if __name__=='__main__':unittest.main()
