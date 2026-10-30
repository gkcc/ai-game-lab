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
from learning.task_single_intervention_v1 import agent_adapter
from learning.task_single_intervention_v2 import prepare, runner


def frame():
    return [{'observation': {'private': {'inventories': [{'A': 1, 'B': 2}], 'shed': {'A': 1, 'B': 2}}}}
            for _ in range(2)]


def target_traces():
    choices = [{'id': 'model', 'unit': 1}, {'id': 'teacher', 'unit': 2}]
    inputs = {'observation': {'step': 2}, 'candidates': choices}
    common = [{'inputs': deepcopy(inputs), 'chosen': 'model', 'teacher_label': 'teacher'} for _ in range(6)]
    changed = deepcopy(common)
    changed[5]['chosen'] = 'teacher'
    before = [{'step': 0, 'records': [], 'crop_records': []},
              {'step': 1, 'records': [], 'crop_records': []}]
    left = before+[{'step': 2, 'records': common, 'crop_records': []}]
    right = deepcopy(before)+[{'step': 2, 'records': changed, 'crop_records': []}]
    gate = {'expected': {'input_sha256': ordered_hash(inputs), 'original_choice': 'model', 'teacher_choice': 'teacher'}}
    return left, right, gate


def stream(rows):
    return io.StringIO(''.join(json.dumps(row, separators=(',', ':'))+'\n' for row in rows))


def crop_record(seconds=0.1):
    return dict(inference_seconds=seconds, chosen='crop', features=[1, 2],
                nested={'inference_seconds': 3.0})


def selection_crop_record(seconds=0.1):
    return dict(step=2, decision=0, unit=1, target=[1, 2], x=[0.25, 0.5],
                schema_sha256='a'*64, teacher='MELON', chosen='MELON', outcome='move_with_seed',
                gate_reason=None, inference_seconds=seconds, logits=[0.1, 0.3],
                prior={'seeds': {'MELON': 1}, 'nested': {'inference_seconds': 3.0}},
                selected_seed_before=1, source='model', unit_request=['NORTH'],
                buy_seed_orders=[['BUY_SEED', 'MELON', 1]])


def matching_proofs(gate, crop_count=0):
    expected = gate['expected']
    before = {'total': {'task_forwards': 5, 'crop_forwards': crop_count},
              'turn': {'task_forwards': 5, 'crop_forwards': crop_count}}
    after = {'total': {'task_forwards': 6, 'crop_forwards': crop_count},
             'turn': {'task_forwards': 6, 'crop_forwards': crop_count}}
    common = dict(step=2, selection_record_index=5, input_sha256=expected['input_sha256'],
                  original_choice='model', teacher_choice='teacher', teacher_label_verified=True,
                  budget_before=before, budget_after=after)
    return (dict(deepcopy(common), mode='control', returned_choice='model'),
            dict(deepcopy(common), mode='intervene', returned_choice='teacher'))


def control_traces():
    before, _, _ = target_traces()
    rows = before+[dict(step=step, records=[], crop_records=[]) for step in range(3, 719)]
    for row in rows:
        row['crop_records'] = [crop_record()]
    return rows


class FakeHistory:
    def read_bytes(self):
        return b'synthetic historical compressed data'


class RunnerTest(unittest.TestCase):
    def test_fixed_case_rejects_old_market_and_wrong_identity(self):
        case = dict(schema='single-task-input-v2', historical_game='holdout-S01-round2-0-merged', side=0, seed=12)
        runner.check_fixed_case(case)
        for changed in (dict(historical_game='holdout-S01-market-0-merged'), dict(side=True),
                        dict(side=1), dict(schema='single-task-input-v1'), dict(seed=True)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runner.check_fixed_case(dict(case, **changed))

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

    def test_runtime_checks_frozen_round2_opponent(self):
        tokens = {}
        class VirtualPath:
            def __init__(self, name):
                self.name = name
            def __truediv__(self, part):
                return VirtualPath(self.name+'/'+str(part).replace('\\', '/'))
            def resolve(self, strict=False):
                return self
            def is_relative_to(self, _other):
                return True
            def is_file(self):
                return True
            def read_bytes(self):
                return tokens[self.name].encode('ascii')
        root, panel, stage = VirtualPath('root'), VirtualPath('panel'), VirtualPath('stage')
        sources = {name:'a'*64 for name in runner.SOURCE_FILES}
        sources['trusted_imports.py'] = runner.TRUSTED_IMPORTS_SHA
        cfg = dict(merged_checkpoint='models/merged', merged_checkpoint_sha256='1'*64,
                   crop_checkpoint='models/crop', crop_checkpoint_sha256='2'*64,
                   engine_sha256='3'*64, official_core_sha256='4'*64, engine_commit='5'*40,
                   source_sha256={'round2.py':'6'*64}, game_limits={})
        private = dict(engine_sha256=cfg['engine_sha256'], engine_commit=cfg['engine_commit'],
                       opponent_sha256='6'*64, source_sha256=sources,
                       wrapper_sha256={'control':'7'*64, 'intervene':'8'*64})
        gate = dict(adapter_source_sha256=sources['agent_adapter.py'],
                    intervention_source_sha256=sources['intervention.py'],
                    crop_sha256=cfg['crop_checkpoint_sha256'], task_sha256=cfg['merged_checkpoint_sha256'],
                    limits=cfg['game_limits'])
        tokens.update({'root/models/merged':'1'*64, 'root/models/crop':'2'*64,
                       'root/learning/task_smoke_weight_probe_v1/core.py':runner.TRACE_VALIDATOR_SHA,
                       'root/learning/panel_eval_v3/launch.py':runner.NUL_LAUNCHER_SHA,
                       'root/.venv/Lib/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py':'3'*64,
                       'root/.venv/Lib/site-packages/kaggle_environments/core.py':'4'*64,
                       'stage/agents/control.py':'7'*64, 'stage/agents/intervene.py':'8'*64,
                       'stage/opponents/round2.py':'6'*64})
        tokens.update({'stage/sources/'+name:digest for name,digest in sources.items()})
        def check(configuration):
            with patch.object(runner, 'read', side_effect=[configuration, {'runtime_files':{}}]), \
                 patch.object(runner, 'sha', side_effect=lambda raw:raw.decode('ascii')):
                return runner.verify_runtime(root, panel, stage, private, gate)
        self.assertIs(cfg, check(cfg))
        bad = dict(cfg, source_sha256={'round2.py':'9'*64})
        with self.assertRaises(ValueError):
            check(bad)

    def test_prefix_accepts_only_fixed_choice_and_common_context(self):
        left, right, gate = target_traces()
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            self.assertEqual({'model_worker': 1, 'teacher_worker': 2},
                             runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash, 0))
        right[2]['records'][4]['chosen'] = 'teacher'
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            with self.assertRaises(ValueError):
                runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash, 0)

    def test_prefix_rejects_changed_input_and_unverified_teacher(self):
        left, right, gate = target_traces()
        right[2]['records'][5]['inputs']['candidates'].append({'id': 'other', 'unit': 3})
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            with self.assertRaises(ValueError):
                runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash, 0)
        left, right, gate = target_traces()
        right[2]['records'][5]['teacher_label'] = 'different'
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            with self.assertRaises(ValueError):
                runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash, 0)

    def test_control_accepts_only_top_level_crop_timing_changes(self):
        left = control_traces()
        right = deepcopy(left)
        for step, row in enumerate(right):
            row['crop_records'][0]['inference_seconds'] = 1.0 + step/1000
        history = FakeHistory()
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            target = runner.trace_control(history, Path('current'), runner.sha(history.read_bytes()), ordered_hash)
        self.assertEqual(2, target['step'])
        self.assertEqual(left[2]['records'][5]['inputs'], target['records'][5]['inputs'])

    def test_control_rejects_business_changes_and_other_timing_paths(self):
        for kind in ('value', 'order', 'timing_key_order', 'record_order', 'nested_timing', 'top_timing',
                     'record_input_timing', 'target_input_timing'):
            left = control_traces()
            right = deepcopy(left)
            if kind == 'value':
                right[10]['crop_records'][0]['features'] = [2, 1]
            elif kind == 'order':
                value = right[10]['crop_records'][0].pop('chosen')
                right[10]['crop_records'][0]['chosen'] = value
            elif kind == 'timing_key_order':
                value = right[10]['crop_records'][0].pop('inference_seconds')
                right[10]['crop_records'][0]['inference_seconds'] = value
            elif kind == 'record_order':
                second = dict(crop_record(), chosen='other')
                left[10]['crop_records'].append(second)
                right[10]['crop_records'].append(deepcopy(second))
                right[10]['crop_records'].reverse()
            elif kind == 'nested_timing':
                right[10]['crop_records'][0]['nested']['inference_seconds'] = 9.0
            elif kind == 'top_timing':
                left[10]['inference_seconds'] = 1.0
                right[10]['inference_seconds'] = 2.0
            elif kind == 'record_input_timing':
                left[10]['records'] = [{'inputs': {'inference_seconds': 1.0}}]
                right[10]['records'] = [{'inputs': {'inference_seconds': 2.0}}]
            else:
                left[2]['records'][5]['inputs']['inference_seconds'] = 1.0
                right[2]['records'][5]['inputs']['inference_seconds'] = 2.0
            history = FakeHistory()
            with self.subTest(kind=kind), patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
                with self.assertRaises(ValueError):
                    runner.trace_control(history, Path('current'), runner.sha(history.read_bytes()), ordered_hash)

    def test_prefix_ignores_only_crop_timing_before_target(self):
        left, right, gate = target_traces()
        for step in (0, 1):
            left[step]['crop_records'] = [crop_record(0.1)]
            right[step]['crop_records'] = [crop_record(0.2)]
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            self.assertEqual({'model_worker': 1, 'teacher_worker': 2},
                             runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash, 0))
        for kind in ('value', 'order', 'timing_key_order', 'record_order', 'nested_timing', 'top_timing',
                     'record_input_timing', 'target_input_timing'):
            left, right, gate = target_traces()
            left[0]['crop_records'] = [crop_record(0.1)]
            right[0]['crop_records'] = [crop_record(0.2)]
            if kind == 'value':
                right[0]['crop_records'][0]['features'] = [2, 1]
            elif kind == 'order':
                value = right[0]['crop_records'][0].pop('chosen')
                right[0]['crop_records'][0]['chosen'] = value
            elif kind == 'timing_key_order':
                value = right[0]['crop_records'][0].pop('inference_seconds')
                right[0]['crop_records'][0]['inference_seconds'] = value
            elif kind == 'record_order':
                second = dict(crop_record(), chosen='other')
                left[0]['crop_records'].append(second)
                right[0]['crop_records'].append(deepcopy(second))
                right[0]['crop_records'].reverse()
            elif kind == 'nested_timing':
                right[0]['crop_records'][0]['nested']['inference_seconds'] = 9.0
            elif kind == 'top_timing':
                left[0]['inference_seconds'] = 1.0
                right[0]['inference_seconds'] = 2.0
            elif kind == 'record_input_timing':
                left[0]['records'] = [{'inputs': {'inference_seconds': 1.0}}]
                right[0]['records'] = [{'inputs': {'inference_seconds': 2.0}}]
            else:
                left[2]['records'][5]['inputs']['inference_seconds'] = 1.0
                right[2]['records'][5]['inputs']['inference_seconds'] = 2.0
            with self.subTest(kind=kind), patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
                with self.assertRaises(ValueError):
                    runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash, 0)

    def test_malformed_crop_record_structure_or_timing_is_rejected(self):
        cases = [{}, {'crop_records': {}}, {'crop_records': [None]},
                 {'crop_records': [{'chosen': 'crop'}]},
                 *[{'crop_records': [dict(inference_seconds=value)]}
                   for value in (True, None, -1, float('inf'), float('nan'))]]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.trace_without_crop_timing(value)
        raw = dict(step=0, crop_records=[crop_record()], inference_seconds=4.0)
        clean = runner.trace_without_crop_timing(raw)
        self.assertEqual(4.0, clean['inference_seconds'])
        self.assertEqual(0.0, clean['crop_records'][0]['inference_seconds'])
        self.assertEqual(list(raw['crop_records'][0]), list(clean['crop_records'][0]))
        self.assertEqual(3.0, clean['crop_records'][0]['nested']['inference_seconds'])
        self.assertIn('inference_seconds', raw['crop_records'][0])
        left, right, gate = target_traces()
        right[2]['crop_records'] = [dict(inference_seconds=True)]
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            with self.assertRaises(ValueError):
                runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash, 0)

    def test_target_crop_prefix_ignores_only_timing_and_late_finish_values(self):
        left, right, gate = target_traces()
        left[2]['crop_records'] = [selection_crop_record(0.1), crop_record(0.3)]
        right[2]['crop_records'] = [selection_crop_record(0.2), dict(crop_record(0.4), chosen='after-target')]
        def check():
            with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
                return runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash, 1)
        self.assertEqual({'model_worker': 1, 'teacher_worker': 2}, check())
        right[2]['crop_records'][0]['unit_request'] = ['SOUTH']
        right[2]['crop_records'][0]['buy_seed_orders'] = []
        right[2]['crop_records'].append(dict(crop_record(0.5), chosen='extra-after-target'))
        self.assertEqual({'model_worker': 1, 'teacher_worker': 2}, check())
        right[2]['crop_records'][0]['chosen'] = 'CARROT'
        with self.assertRaises(ValueError):
            check()

    def test_target_crop_prefix_schema_and_decision_fields_fail_closed(self):
        for kind in ('unknown', 'missing_select', 'missing_finish', 'reordered', 'nested_timing', 'timing_order'):
            left, right, gate = target_traces()
            left[2]['crop_records'] = [selection_crop_record(0.1)]
            right[2]['crop_records'] = [selection_crop_record(0.2)]
            record = right[2]['crop_records'][0]
            if kind == 'unknown':
                record['unrecognized'] = True
            elif kind == 'missing_select':
                del record['chosen']
            elif kind == 'missing_finish':
                del record['unit_request']
            elif kind == 'reordered':
                value = record.pop('chosen')
                record['chosen'] = value
            elif kind == 'nested_timing':
                record['prior']['nested']['inference_seconds'] = 9.0
            else:
                value = record.pop('inference_seconds')
                record['inference_seconds'] = value
            with self.subTest(kind=kind), patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
                with self.assertRaises(ValueError):
                    runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash, 1)

    def test_invalid_or_overlong_target_crop_prefix_count_is_rejected(self):
        left, right, gate = target_traces()
        left[2]['crop_records'] = [selection_crop_record()]
        right[2]['crop_records'] = [selection_crop_record()]
        for count in (True, False, -1, 1.0, None, '1', 2):
            with self.subTest(count=count), patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
                with self.assertRaises(ValueError):
                    runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash, count)
        a, b = matching_proofs(gate, crop_count=2)
        with patch.object(runner, 'read', side_effect=[a, b]):
            count = runner.check_proofs(Path('stage'), gate)
        with patch.object(runner.gzip, 'open', side_effect=[stream(left), stream(right)]):
            with self.assertRaises(ValueError):
                runner.trace_prefix(Path('a'), Path('b'), gate, ordered_hash, count)

    def test_proofs_require_same_model_budget_and_single_return_change(self):
        _, _, gate = target_traces()
        a, b = matching_proofs(gate)
        with patch.object(runner, 'read', side_effect=[a, b]):
            self.assertEqual(0, runner.check_proofs(Path('stage'), gate))
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

    def test_proofs_reject_invalid_crop_count_even_when_both_match(self):
        _, _, gate = target_traces()
        for value in (True, False, -1, 1.0, None, '1'):
            a, b = matching_proofs(gate)
            a['budget_before']['turn']['crop_forwards'] = value
            b['budget_before']['turn']['crop_forwards'] = value
            with self.subTest(value=value), patch.object(runner, 'read', side_effect=[a, b]):
                with self.assertRaises(ValueError):
                    runner.check_proofs(Path('stage'), gate)

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
