import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch, mock_open
from contextlib import ExitStack

impact = types.ModuleType('learning.task_action_impact_diagnostic_v1.analysis')
def compare(left,right):
    if not left or not right or left[0]['inputs'] != right[0]['inputs']:
        return dict(choice=None,context_stop='input_difference')
    if left[0]['chosen'] == right[0]['chosen']: return dict(choice=None,context_stop=None)
    return dict(choice=dict(callback=0,candidate_op=left[0]['op'],reference_op=right[0]['op'],same_unit=left[0]['unit']==right[0]['unit'],emitted_unit_request_equal=False),context_stop=None)
impact.compare_callbacks = compare
impact.request_digest = lambda value: repr(value)
for name in ('Meter','compact_replay','descriptor','first_request_difference','read_json','trace_lines','validate_trace'):
    setattr(impact,name,lambda *a,**kw: None)
observation = types.ModuleType('learning.task_action_observation_v1.analysis'); observation.raw_path = lambda *a: None
collect = types.ModuleType('learning.task_completion_bc_v1.collect'); collect.current_game = lambda *a: None
experiment = types.ModuleType('learning.task_completion_worst_pair_v1.experiment')
experiment.OUT = Path(__file__).resolve().parents[2]/'synthetic'
experiment.PARENT = experiment.OUT.parent/'synthetic-parent'
for name in ('deadline_check','read','ready','sha','verify_commit','write_new'): setattr(experiment,name,lambda *a: None)
spec = importlib.util.spec_from_file_location('learning.task_completion_worst_pair_v1._analysis_test',Path(__file__).with_name('analysis.py'))
a = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'learning.task_action_impact_diagnostic_v1.analysis':impact,
    'learning.task_action_observation_v1.analysis':observation,'learning.task_completion_bc_v1.collect':collect,
    'learning.task_completion_worst_pair_v1.experiment':experiment}): spec.loader.exec_module(a)


class AnalysisTest(unittest.TestCase):
    def test_choice_boundaries(self):
        left = [dict(inputs='same',chosen='a',op='DIG',unit=0,unit_request=['DIG'])]
        right = [dict(inputs='same',chosen='b',op='WATER',unit=1,unit_request=['WATER'])]
        for different in (False,True):
            summary, unit = a.choice_summary(left,right,different)
            self.assertEqual(summary['complete_request_different'],different)
            self.assertFalse(summary['emitted_unit_request_equal'])
            self.assertEqual(unit,(0,repr(['DIG'])))
        self.assertIsNone(a.choice_summary(left,[{**right[0],'inputs':'other'}],True)[0])
        self.assertIsNone(a.choice_summary(left,[{**right[0],'chosen':'a'}],True)[0])

    def test_events(self):
        self.assertEqual(a.event_summary([]),{'observation':'unknown'})
        for status,changed in (('blocked',False),('executed',True)):
            event = dict(requested=['DIG'],effective=['DIG'],status=status,before=0,after=int(changed),tile_before=0,tile_after=0)
            result = a.event_summary([event])
            self.assertEqual(result['status'],status)
            self.assertEqual(result['atomic_change_observed'],changed)
        self.assertEqual(a.event_summary([{},{}]),{'observation':'unknown'})
        for direction in ('NORTH','SOUTH','EAST','WEST'):
            result = a.event_summary([dict(requested=[direction],effective=[direction],status='executed')])
            self.assertEqual(result['requested_operation'],direction)
            self.assertEqual(a.event_summary([dict(requested=[direction],effective=[direction],status='executed')],repr([direction]))['observation'],'unique')
            self.assertEqual(a.event_summary([dict(requested=[direction],effective=[direction],status='executed')],repr(['DIG']))['observation'],'unknown')
        for event in (dict(requested=['DIG']),dict(effective=['DIG']),dict()):
            self.assertIsNone(a.event_summary([event])['rewritten'])

    def test_returned_replay_without_count_is_unknown(self):
        self.assertEqual(a.transition_progress(True,None),dict(confirmed_transitions='unknown',partial_replay_transitions='unknown'))
        self.assertEqual(a.transition_progress(False,None),dict(confirmed_transitions=0,partial_replay_transitions=0))
        self.assertEqual(a.transition_progress(True,719),dict(confirmed_transitions=719,partial_replay_transitions=0))

    def test_raw_identity_mismatch_rejected(self):
        value = dict(path='games/example.gz',size=10,sha256='a'*64)
        games = {arm:{kind:dict(value) for kind in ('replay','trace')} for arm in ('candidate','teacher_task')}
        registered = {arm:{kind:dict(value) for kind in ('replay','trace')} for arm in games}
        a.validate_raw_identities(games,registered)
        for field, replacement in (('path','other.gz'),('size',11),('sha256','b'*64)):
            changed = {arm:{kind:dict(item) for kind,item in entries.items()} for arm,entries in registered.items()}
            changed['candidate']['trace'][field] = replacement
            with self.assertRaises(ValueError): a.validate_raw_identities(games,changed)

    def test_partial_marker_and_receipt_failure(self):
        for failure in ('partial','receipt'):
            with self.subTest(failure=failure), ExitStack() as stack:
                records = {}
                raw = {arm:{kind:dict(path='x',size=1,sha256='a'*64) for kind in ('replay','trace')} for arm in ('candidate','teacher_task')}
                cfg = dict(seed=a.SEED,side=0,opponent='round2',candidate_checkpoint_sha256='x',crop_checkpoint_sha256='y',engine_sha256='z',raw_inputs=raw,
                    limits=dict(compressed_bytes=128*1024**2,decoded_bytes=1024**3,seconds=300,official_transitions=719,raw_files=4,decode_passes_per_file=1))
                rows = [dict(seed=a.SEED,side=0,opponent='round2',policy=p) for p in ('candidate','teacher_task')]
                def read(path): return dict(passed=True,game_ledger_sha256='h',audit_ledger_sha256='h') if Path(path).name=='games-result.json' else rows
                def write(path,value):
                    if failure == 'receipt': raise PermissionError()
                    records[Path(path).name] = value
                decision = types.ModuleType('decision'); full = types.ModuleType('decision.full_audit'); full.reexecute = lambda *a,**kw: None
                stack.enter_context(patch.dict(sys.modules,{'decision':decision,'decision.full_audit':full}))
                for name,value in (('ready',cfg),('verify_commit',None),('sha','h'),('deadline_check',None)):
                    stack.enter_context(patch.object(a,name,return_value=value))
                stack.enter_context(patch.object(a,'read',side_effect=read))
                stack.enter_context(patch.object(a,'current_game',return_value=(raw['candidate'],{})))
                stack.enter_context(patch.object(a,'write_new',side_effect=write))
                stack.enter_context(patch.object(Path,'exists',return_value=False))
                stack.enter_context(patch.object(Path,'open',mock_open()))
                if failure == 'partial': stack.enter_context(patch.object(a.json,'dump',side_effect=OSError()))
                else: stack.enter_context(patch.object(a,'Meter',side_effect=ValueError()))
                with self.assertRaises(Exception) as caught: a.run()
                if failure == 'partial': self.assertFalse(records['analysis-failure.local.json']['marker']['complete'])
                else:
                    self.assertEqual(caught.exception.primary_error_kind,'ValueError')
                    self.assertEqual(caught.exception.receipt_error_kind,'PermissionError')


if __name__ == '__main__': unittest.main()
