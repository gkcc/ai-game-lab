"""Pure in-memory data and evaluation failure receipts."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch, mock_open
from contextlib import ExitStack
from .test_entry import c

core = types.ModuleType('learning.task_completion_bc_v1.collect')
core.ready = core.write_new = core.deadline_check = lambda *a: None
spec = importlib.util.spec_from_file_location('learning.task_completion_bc_reentry_v1._collect_test_instance', Path(__file__).with_name('collect.py'))
collect = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'learning.task_completion_bc_v1.collect': core}):
    spec.loader.exec_module(collect)
# If another test imported the real package first, its cached package attribute
# can win over sys.modules. Bind the private test instance explicitly.
collect.core = core


class DataEntryTest(unittest.TestCase):
    def test_extraction_failure_and_replay_restoration(self):
        for case in ('partial', 'init', 'replay', 'receipt'):
            with self.subTest(case=case):
                out = collect.experiment.OUT
                records = {}
                full = types.ModuleType('decision.full_audit')
                def original(*args, **kwargs):
                    raise LookupError('interrupted replay')
                full.reexecute = original
                def write(path, value):
                    if case == 'receipt': raise PermissionError('receipt')
                    records[Path(path).name] = value
                def run(_):
                    core.write_new(out/'test-data-started.json', {'attempt':1})
                    if case in ('init', 'receipt'): raise ValueError('initialization')
                    if case == 'replay': full.reexecute(None)
                core.test = run
                campaign = types.ModuleType('learning.task_completion_bc_reentry_v1.campaign')
                campaign.verify_model_seal = lambda _: None
                decision = types.ModuleType('decision')
                decision.full_audit = full
                with ExitStack() as stack:
                    stack.enter_context(patch.dict(sys.modules, {'learning.task_completion_bc_reentry_v1.campaign':campaign,
                        'decision':decision, 'decision.full_audit':full}))
                    stack.enter_context(patch.object(collect.experiment,'write_new',side_effect=write))
                    stack.enter_context(patch.object(Path,'open',mock_open()))
                    if case == 'partial': stack.enter_context(patch.object(collect.json,'dump',side_effect=OSError('partial')))
                    with self.assertRaises(Exception) as caught: collect.test(out)
                self.assertIs(full.reexecute, original)
                if case == 'receipt':
                    self.assertEqual(caught.exception.primary_error_kind, 'ValueError')
                    self.assertEqual(caught.exception.receipt_error_kind, 'PermissionError')
                else:
                    receipt = records['test-data-entry-failure.local.json']
                    if case == 'partial': self.assertFalse(receipt['marker_complete'])
                    if case == 'replay':
                        self.assertEqual(receipt['replay_started'], 1)
                        self.assertEqual(receipt['replay_completed'], 0)
                        self.assertEqual(receipt['current_replay_partial_transitions'], 'unknown')

    def test_evaluate_marker_and_data_failure(self):
        for case in ('partial', 'data', 'receipt', 'retry'):
            with self.subTest(case=case), ExitStack() as stack:
                out = c.OUT
                records = {}
                stack.enter_context(patch.object(c,'ready',return_value={'data_limits':{}}))
                stack.enter_context(patch.object(c,'verify_commit'))
                stack.enter_context(patch.object(c,'verify_model_seal',return_value=({'checkpoint_sha256':'hash'},{'baseline':{}})))
                stack.enter_context(patch.object(c,'verify_independent_review',return_value={}))
                stack.enter_context(patch.object(c,'read',return_value=dict(passed=True, model_seal_sha256='hash', games_result_sha256='hash', label_ledger_sha256='hash', array_sha256='expected')))
                stack.enter_context(patch.object(c,'sha',return_value='hash'))
                stack.enter_context(patch.object(Path,'exists',return_value=(case=='retry')))
                stack.enter_context(patch.object(Path,'open',mock_open()))
                def write(path, value):
                    if case == 'receipt': raise PermissionError()
                    records[Path(path).name] = value
                stack.enter_context(patch.object(c,'write_new',side_effect=write))
                if case == 'partial': stack.enter_context(patch.object(c.json,'dump',side_effect=OSError()))
                with self.assertRaises(Exception) as caught: c.evaluate(out)
                if case == 'receipt': self.assertEqual(caught.exception.receipt_error_kind,'PermissionError')
                if case == 'partial': self.assertFalse(records['evaluation-failure.local.json']['marker_complete'])
                if case == 'data': self.assertTrue(records['evaluation-failure.local.json']['test_data_read_started'])
                if case == 'retry': self.assertNotIn('evaluation-failure.local.json', records)


if __name__ == '__main__': unittest.main()
