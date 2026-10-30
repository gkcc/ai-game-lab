"""Synthetic regressions for the one-case comparison boundary."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from decision.semantics import canonical
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_single_intervention_v1 import agent_adapter, prepare, runner


def frame():
    return [{'observation': {'private': {'inventories': [{'A': 1, 'B': 2}], 'shed': {'A': 1, 'B': 2}}}}
            for _ in range(2)]


def target_traces():
    choices = [{'id': 'model', 'unit': 1}, {'id': 'teacher', 'unit': 2}]
    inputs = {'observation': {'step': 2}, 'candidates': choices}
    common = [{'inputs': inputs, 'chosen': 'model', 'teacher_label': 'teacher'} for _ in range(6)]
    changed = deepcopy(common)
    changed[5]['chosen'] = 'teacher'
    before = [{'step': 0, 'records': []}, {'step': 1, 'records': []}]
    left = before+[{'step': 2, 'records': common}]
    right = deepcopy(before)+[{'step': 2, 'records': changed}]
    gate = {'expected': {'input_sha256': ordered_hash(inputs), 'original_choice': 'model', 'teacher_choice': 'teacher'}}
    return left, right, gate


def stream(rows):
    return io.StringIO(''.join(json.dumps(row, separators=(',', ':'))+'\n' for row in rows))


class RunnerTest(unittest.TestCase):
    def test_registered_public_runtime_passes_real_dependency_preflight(self):
        source = str(Path(__file__).resolve().parent/'prepare.py')
        code = ("import runpy,sys,pathlib; d=runpy.run_path("+repr(source)+",run_name='preflight'); "
                "a,h,p=d['dependencies'](); assert callable(a.trace_lines) and callable(h) and callable(p.checked_rows); "
                "assert not pathlib.Path(sys.pycache_prefix).exists()")
        result = subprocess.run([sys.executable, '-B', '-I', '-c', code], capture_output=True, text=True, timeout=30)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_preloaded_project_module_is_rejected_before_import(self):
        source = str(Path(__file__).resolve().parent/'prepare.py')
        code = ("import runpy,sys,types; sys.modules['learning.task_bc_error_diagnostics_v1.shadow']=types.ModuleType('fake'); "
                "d=runpy.run_path("+repr(source)+",run_name='preflight'); d['dependencies']()")
        result = subprocess.run([sys.executable, '-B', '-I', '-c', code], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(0, result.returncode)

    def test_target_extraction_uses_the_recorded_teacher_candidate(self):
        choices = [{'id':'model','op':'PLANT','target':[1, 2],'unit':1},
                   {'id':'teacher','op':'PLANT','target':[1, 2],'unit':2}]
        inputs = {'observation': {'step': 2}, 'candidates': choices}
        record = dict(source='callback', inputs=inputs, chosen='model', teacher_label='teacher')
        class FakeAnalysis:
            class Meter:
                def __init__(self, **_kwargs):
                    self.receipts = []
            @staticmethod
            def trace_lines(_path, _identity, meter):
                for step in range(719):
                    yield dict(step=step, checkpoint_sha256='crop', task_checkpoint_sha256='task',
                               load_errors=0, inference_errors=0, fallbacks=0,
                               records=[record]*6 if step == 2 else [])
                meter.receipts.append({'passed': True})
        game = {'trace': {'path':'trace'}, 'row': {'crop_checkpoint_sha256':'crop','task_checkpoint_sha256':'task'}}
        target = prepare.target_from_trace(Path('panel'), game, FakeAnalysis, ordered_hash)
        self.assertEqual({'input_sha256':ordered_hash(inputs),'original_choice':'model','teacher_choice':'teacher'}, target)
        choices[1]['target'] = [9, 9]
        with self.assertRaises(ValueError):
            prepare.target_from_trace(Path('panel'), game, FakeAnalysis, ordered_hash)

    def test_comparison_preserves_shed_and_inventory_order(self):
        original = frame()
        changed = deepcopy(original)
        changed[1]['observation']['private']['shed'] = {'B': 2, 'A': 1}
        self.assertEqual(canonical(original), canonical(changed))
        self.assertFalse(runner.same_frame(original, changed, canonical))
        changed = deepcopy(original)
        changed[0]['observation']['private']['inventories'] = [{'B': 2, 'A': 1}]
        self.assertFalse(runner.same_frame(original, changed, canonical))
        self.assertTrue(runner.same_frame(original, deepcopy(original), canonical))

    def test_source_manifest_cannot_silently_drop_a_file(self):
        sources = {name:'a'*64 for name in runner.SOURCE_FILES}
        sources['trusted_imports.py'] = runner.TRUSTED_IMPORTS_SHA
        private = {'source_sha256':sources, 'wrapper_sha256':{'control':'b'*64,'intervene':'c'*64}}
        gate = {'adapter_source_sha256':'a'*64,'intervention_source_sha256':'a'*64}
        runner.check_source_manifest(private, gate)
        incomplete = deepcopy(private)
        del incomplete['source_sha256']['runner.py']
        with self.assertRaises(ValueError):
            runner.check_source_manifest(incomplete, gate)
        with self.assertRaises(ValueError):
            runner.check_source_manifest(private, dict(gate, adapter_source_sha256='d'*64))

    def test_prefix_accepts_only_fixed_choice_and_common_context(self):
        left, right, gate = target_traces()
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            self.assertEqual({'model_worker': 1, 'teacher_worker': 2},
                             runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash))
        right[2]['records'][4]['chosen'] = 'teacher'
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            with self.assertRaises(ValueError):
                runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash)

    def test_prefix_rejects_changed_input_and_unverified_teacher(self):
        left, right, gate = target_traces()
        right[2]['records'][5]['inputs']['candidates'].append({'id': 'other', 'unit': 3})
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            with self.assertRaises(ValueError):
                runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash)
        left, right, gate = target_traces()
        right[2]['records'][5]['teacher_label'] = 'different'
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            with self.assertRaises(ValueError):
                runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash)

    def test_proofs_require_same_model_budget_and_single_return_change(self):
        _, _, gate = target_traces()
        expected = gate['expected']
        common = dict(step=2, selection_record_index=5, input_sha256=expected['input_sha256'],
                      original_choice='model', teacher_choice='teacher', teacher_label_verified=True,
                      budget_before={'task_forwards': 5}, budget_after={'task_forwards': 6})
        a = dict(common, mode='control', returned_choice='model')
        b = dict(common, mode='intervene', returned_choice='teacher')
        with patch.object(runner, 'read', side_effect=[a, b]):
            runner.check_proofs(Path('stage'), gate)
        agent_adapter._verify_control(a, b)
        b['budget_after'] = {'task_forwards': 7}
        with patch.object(runner, 'read', side_effect=[a, b]):
            with self.assertRaises(ValueError):
                runner.check_proofs(Path('stage'), gate)
        with self.assertRaises(ValueError):
            agent_adapter._verify_control(a, b)
        a['budget_after'] = {'task_forwards': 1}
        b['budget_after'] = {'task_forwards': True}
        with patch.object(runner, 'read', side_effect=[a, b]):
            with self.assertRaises(ValueError):
                runner.check_proofs(Path('stage'), gate)
        with self.assertRaises(ValueError):
            agent_adapter._verify_control(a, b)

    def test_official_effect_summary_retains_worker_roles(self):
        class FakeStage:
            def __truediv__(self, _):
                return self
            def open(self, *_args, **_kwargs):
                return io.StringIO()
        def fake_reexecute(_replay, trace, label_sink):
            self.assertTrue(trace)
            label_sink(dict(phase='unit', player=0, step=2, unit=2, status='no_effect',
                            requested=['EAST'], effective=['EAST']))
            label_sink(dict(phase='unit', player=0, step=2, unit=1, status='executed',
                            requested=['NORTH'], effective=['NORTH']))
            return dict(passed=True, official_transitions_reexecuted=719,
                        both_public_and_private_states_match_each_step=True, inventory_order_checked_each_step=True)
        result = runner.official_audit(FakeStage(), 'control', {},
                                       {'model_worker': 1, 'teacher_worker': 2}, fake_reexecute)
        self.assertEqual({'status':'executed','requested_op':'NORTH','requested_equals_effective':True},
                         result['roles']['model_worker'])
        self.assertEqual({'status':'no_effect','requested_op':'EAST','requested_equals_effective':True},
                         result['roles']['teacher_worker'])

    def test_control_proof_budget_is_valid_before_intervention_starts(self):
        _, _, gate = target_traces()
        expected = gate['expected']
        fields = ('task_forwards','task_rows','crop_forwards','crop_rows',
                  'gate_evaluations','controller_calls','teacher_proposals')
        before = {key: 0 for key in fields}
        after = dict(before, task_forwards=1, task_rows=2)
        record = dict(mode='control', step=2, selection_record_index=5, teacher_label_verified=True,
                      input_sha256=expected['input_sha256'], original_choice='model', returned_choice='model',
                      teacher_choice='teacher', budget_before={'total': before, 'turn': dict(before)},
                      budget_after={'total': after, 'turn': dict(after)})
        with patch.object(runner, 'read', return_value=record):
            runner.check_control_proof(Path('stage'), gate, 2)
        record['budget_after']['total']['task_forwards'] = True
        with patch.object(runner, 'read', return_value=record):
            with self.assertRaises(ValueError):
                runner.check_control_proof(Path('stage'), gate, 2)

    def test_invalid_paths_do_not_echo_private_arguments(self):
        source = Path(__file__).resolve().parent
        marker = 'SYNTHETIC_PRIVATE_PATH_SENTINEL'
        commands = [
            [str(source/'prepare.py'), '--panel', marker, '--status', marker, '--game-root', marker, '--stage', marker],
            [str(source/'runner.py'), '--panel', marker, '--game-root', marker, '--stage', marker],
        ]
        for arguments in commands:
            result = subprocess.run([sys.executable, '-B', '-I', *arguments], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(0, result.returncode)
            self.assertNotIn(marker, result.stdout+result.stderr)
            self.assertNotIn('Traceback', result.stdout+result.stderr)

    @unittest.skipUnless(os.name == 'nt', 'Windows Job required')
    def test_synthetic_windows_job_uses_nul_standard_handles(self):
        from learning.task_job_empty_exit_v1.sampling import supervise
        command = [sys.executable, '-B', '-I', '-c',
                   "import ctypes,os,time; k=ctypes.WinDLL('kernel32'); k.GetStdHandle.restype=ctypes.c_void_p; "
                   "k.GetFileType.argtypes=[ctypes.c_void_p]; "
                   "assert all(k.GetFileType(k.GetStdHandle(n)) == 2 for n in (-10,-11,-12)); "
                   "os.write(1,b'SYNTHETIC_NUL_MARKER'); os.write(2,b'SYNTHETIC_NUL_MARKER'); time.sleep(.4)"]
        limits = dict(max_rss_bytes=2*2**30, max_commit_bytes=3*2**30, min_available_bytes=128*2**20,
                      wall_seconds=15, poll_seconds=.05, cleanup_seconds=5)
        result = supervise(runner.nul_api(), command, Path.cwd(), limits)
        self.assertTrue(result['passed'])
        self.assertTrue(result['launcher_reaped'] and result['job_members_zero'])


if __name__ == '__main__':
    unittest.main()
