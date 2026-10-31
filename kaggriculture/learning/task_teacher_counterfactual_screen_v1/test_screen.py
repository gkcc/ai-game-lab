"""Pure synthetic tests; never read historical game data or run an engine."""
from copy import deepcopy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_student_continuation_v1.compare import PrivateComparison
from learning.task_student_dagger_collection_v1 import sampling
from learning.task_student_collection_v1.test_sampling import TASK,CROP,choice,trace
from . import screen


def candidate(step, ordinal=0):
    record = dict(source='callback', chosen='student-private', teacher_label='teacher-private',
                  inputs=dict(candidates=[dict(id='student-private'),dict(id='teacher-private')]))
    doc = dict(step=step,records=[record] * (ordinal + 1))
    return dict(step=step,ordinal=ordinal,third_candidate=None,document=doc,previous_document=None)


def sorted_window(steps):
    values = [candidate(step) for step in steps]
    return sorted(values,key=lambda x:(sampling.coordinate_rank('T01','market',0,x['step'],x['ordinal']),
                                       x['step'],x['ordinal']))


def effect(changed):
    return dict(factual_reproduced=True,request_changed=True,state_changed=changed,
                scope='one_official_transition',complete_game=False,future_return_evaluated=False)


class FakeGame:
    identity = dict(source_role='A05',batch=2,role='T01',opponent='market',side=0)
    task_sha256, crop_sha256, game_limits = 'a'*64, 'b'*64, {}
    def __init__(self, windows):
        self.windows = windows
        self.replay = dict(steps=[[{'action':{}},{'action':{}}] for _ in range(720)],configuration={})
    def scan_candidates(self, *, guard):
        guard()
        return {'lines':719}, self.windows
    def make_agent(self):
        raise AssertionError('Synthetic comparer owns its fake agent')


class ScreeningTests(unittest.TestCase):
    def view(self):
        return patch.object(screen,'observe',return_value=SimpleNamespace(observation={},configuration={}))

    def test_fixed_window_rank_stop_after_first_next_frame_difference(self):
        windows = [sorted_window([1,3,8,10]), sorted_window([121,124]), sorted_window([242]), [], [], []]
        game = FakeGame(windows)
        seen, operations = [], []
        def compare(doc,*_args,**_kwargs):
            seen.append(doc['step'])
            i = len(seen)
            if i == 1: return PrivateComparison('same','same',{'request_equal':True})
            if i == 2: return PrivateComparison('unknown','synthetic')
            return PrivateComparison('different','different',{'request_equal':False},
                {'farmer':['PASS'],'hands':[],'market':[]},
                {'farmer':['DIG'],'hands':[],'market':[]})
        def official(_replay,**_kwargs):
            operations.append(None)
            return effect(len(operations) == 2)
        with self.view(), patch.object(screen,'observation',return_value={}):
            result = screen.run_case(game,role='A05',opponent='market',side=0,guard=lambda:None,
                                     compare_turn=compare,compare_step=official,meter=screen.Meter())
        self.assertEqual(len(seen),4)
        self.assertEqual(result.coverage['status_by_window'][0], ['request_same','unknown'])
        self.assertEqual(result.coverage['status_by_window'][1],
            ['official_next_frame_no_difference_detected','official_next_frame_different'])
        self.assertEqual(result.coverage['status_by_window'][2], ['unexamined_after_hit','empty'])
        self.assertEqual(result.coverage['ranked_out_by_window'],[2,0,0,0,0,0])
        self.assertEqual(result.coverage['source_disagreements_beyond_top4'],'unknown')
        self.assertEqual((result.coverage['comparisons'],result.coverage['official_transitions_reserved']),(4,4))
        self.assertTrue(result.coverage['candidate_found'])
        self.assertNotIn('student-private',repr(result))

    def test_structural_only_difference_does_not_call_official(self):
        windows = [sorted_window([1]),[],[],[],[],[]]
        game = FakeGame(windows)
        left = {'farmer':['PASS'],'hands':[],'market':[]}
        right = {'market':[],'hands':[],'farmer':['PASS']}
        comparison = PrivateComparison('different','structural',{'request_equal':False},left,right)
        with self.view(), patch.object(screen,'observation',return_value={}):
            result = screen.run_case(game,role='A05',opponent='market',side=0,guard=lambda:None,
                compare_turn=lambda *_a,**_k:comparison,
                compare_step=lambda *_a,**_k:self.fail('official executor must not run'),meter=screen.Meter())
        self.assertEqual(result.coverage['status_by_window'][0][0],'official_request_same')
        self.assertEqual(result.coverage['official_transitions_reserved'],0)
        self.assertIsNone(result.candidate)

    def test_rejects_unordered_duplicate_wrong_role_and_bad_receipt(self):
        for kind in ('unordered','duplicate','role','receipt'):
            windows = [sorted_window([1,3]),[],[],[],[],[]]
            game = FakeGame(windows)
            if kind == 'unordered': windows[0].reverse()
            elif kind == 'duplicate': windows[0][1] = deepcopy(windows[0][0])
            elif kind == 'role': game.identity = dict(game.identity,source_role='A06')
            comparison = PrivateComparison('different','different',{'request_equal':False},
                {'farmer':['PASS'],'hands':[],'market':[]},{'farmer':['DIG'],'hands':[],'market':[]})
            with self.subTest(kind=kind), self.view(), patch.object(screen,'observation',return_value={}):
                with self.assertRaises(ValueError):
                    screen.run_case(game,role='A05',opponent='market',side=0,guard=lambda:None,
                        compare_turn=lambda *_a,**_k:comparison,
                        compare_step=lambda *_a,**_k:dict(effect(True),factual_reproduced=False),meter=screen.Meter())

    def test_failed_helper_preserves_owner_meter_and_does_not_claim_two_steps(self):
        game, meter = FakeGame([sorted_window([1]),[],[],[],[],[]]), screen.Meter()
        comparison = PrivateComparison('different','different',{'request_equal':False},
            {'farmer':['PASS'],'hands':[],'market':[]},{'farmer':['DIG'],'hands':[],'market':[]})
        def failed(*_args,**_kwargs): raise MemoryError('synthetic stop')
        with self.view(), patch.object(screen,'observation',return_value={}):
            with self.assertRaises(MemoryError):
                screen.run_case(game,role='A05',opponent='market',side=0,guard=lambda:None,
                    compare_turn=lambda *_a,**_k:comparison,compare_step=failed,meter=meter)
        counts = meter.snapshot()
        self.assertEqual((counts['comparisons_reserved'],counts['comparisons_begun'],
                          counts['comparisons_completed']),(1,1,1))
        self.assertEqual((counts['official_transitions_reserved'],counts['official_helpers_begun'],
                          counts['official_helpers_completed'],counts['official_transitions_confirmed_completed']),
                         (2,1,0,0))
        self.assertEqual(counts['incomplete_helper_step_progress'],'unknown')

    def test_real_full_synthetic_source_scanner_connects_both_seats(self):
        for role,slot,opponent,side in (('A05','T01','market',0),('A06','T02','round2',1)):
            raw=trace({1:[choice()],125:[choice()]})
            scanner=sampling.RankedStream(io.BytesIO(raw),role=slot,
                                          opponent=opponent,side=side,guard=lambda:None)
            self.assertEqual(scanner.scan(task_sha256=TASK,crop_sha256=CROP)['lines'],719)
            game=FakeGame(scanner.candidates())
            game.identity=dict(source_role=role,batch=2,role=slot,opponent=opponent,side=side)
            comparison=PrivateComparison('different','different',{'request_equal':False},
                {'farmer':['PASS'],'hands':[],'market':[]},{'farmer':['DIG'],'hands':[],'market':[]})
            meter=screen.Meter()
            with self.subTest(role=role),self.view(),patch.object(screen,'observation',return_value={}):
                result=screen.run_case(game,role=role,opponent=opponent,side=side,guard=lambda:None,
                    compare_turn=lambda *_a,**_k:comparison,compare_step=lambda *_a,**_k:effect(True),meter=meter)
            self.assertTrue(result.coverage['candidate_found'])
            self.assertEqual(result.candidate['role'],role)
            self.assertEqual(result.coverage['comparisons'],1)
            self.assertEqual(meter.snapshot()['official_transitions_confirmed_completed'],2)


if __name__ == '__main__': unittest.main()
