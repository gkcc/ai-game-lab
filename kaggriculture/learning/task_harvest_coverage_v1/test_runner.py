"""Metadata coordination invariants without private files or processes."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import runner


class Stream:
    def __init__(self, values, *, close_error=False):
        self.values = iter(values)
        self.closed = False
        self.close_error = close_error

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.values)

    def close(self):
        self.closed = True
        if self.close_error:
            raise OSError('synthetic cleanup failure')


def run_stream(stream, callback):
    scan = dict(rows=0, eof=False, closed=False, close_errors=0)
    return runner.consume_trace(stream, callback, side=0, first_days={}, guard=lambda:True, scan=scan), scan


class RunnerTests(unittest.TestCase):
    def test_consumer_must_reach_eof_not_only_read_719_rows(self):
        result, scan = run_stream(Stream(range(719)), lambda rows, **_:list(rows))
        self.assertEqual(len(result),719)
        self.assertEqual(scan,dict(rows=719,eof=True,closed=True,close_errors=0))
        def early(rows, **_):
            return [next(rows) for _ in range(719)]
        with self.assertRaises(ValueError):
            run_stream(Stream(range(719)),early)

    def test_extra_and_missing_rows_reject(self):
        for n in (718,720):
            with self.assertRaises(ValueError):
                run_stream(Stream(range(n)),lambda rows, **_:list(rows))

    def test_cleanup_error_cannot_mask_primary_or_turn_success_into_pass(self):
        def failed(_rows, **_):
            raise MemoryError('original resource failure')
        with self.assertRaisesRegex(MemoryError,'original resource failure'):
            run_stream(Stream(range(719),close_error=True),failed)
        with self.assertRaisesRegex(OSError,'synthetic cleanup failure'):
            run_stream(Stream(range(719),close_error=True),lambda rows, **_:list(rows))

    def test_aggregate_keeps_denominators_and_rejects_noninteger_or_missing_counts(self):
        api = SimpleNamespace(empty_counts=lambda:{'trace':{'rows':0},'callbacks':{'total':0}})
        values = [dict(counts={'trace':{'rows':719},'callbacks':{'total':i}}) for i in range(4)]
        self.assertEqual(runner.aggregate(values,api),{'trace':{'rows':2876},'callbacks':{'total':6}})
        for bad in (True,-1,1.0):
            altered=deepcopy(values);altered[0]['counts']['callbacks']['total']=bad
            with self.assertRaises(ValueError):runner.aggregate(altered,api)
        altered=deepcopy(values);del altered[0]['counts']['callbacks']
        with self.assertRaises(ValueError):runner.aggregate(altered,api)
        with self.assertRaises(ValueError):runner.aggregate(values[:3],api)

    def test_cli_help_no_run_and_errors_do_not_echo_arguments(self):
        out,err=io.StringIO(),io.StringIO()
        with redirect_stdout(out),redirect_stderr(err),patch.object(runner,'prepare') as prepare,patch.object(runner,'run') as run:
            self.assertEqual(runner.main(['--help']),0)
            prepare.assert_not_called();run.assert_not_called()
            self.assertEqual(runner.main(['--unknown','SYNTHETIC_PRIVATE_ARGUMENT']),1)
            self.assertEqual(runner.main(['--prepare','--registration-commit','a'*40]),1)
            run.side_effect=SystemExit(0)
            self.assertEqual(runner.main(['--run','--registration-commit','a'*40]),1)
        self.assertNotIn('SYNTHETIC_PRIVATE_ARGUMENT',out.getvalue()+err.getvalue())

    def test_final_receipt_writer_preserves_original_error(self):
        def fail(*_args,**_kwargs):
            raise OSError('synthetic receipt failure')
        infra=SimpleNamespace(write=fail)
        with self.assertRaisesRegex(MemoryError,'original'):
            try:raise MemoryError('original')
            finally:
                import sys
                runner.close(infra,'unused',{},sys.exc_info()[1])
        with self.assertRaises(OSError):runner.close(infra,'unused',{},None)


if __name__=='__main__':
    unittest.main()
