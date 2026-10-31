"""Synthetic integration and accounting checks; no historical inputs."""
from contextlib import contextmanager
from copy import deepcopy
import io
from types import SimpleNamespace
from unittest import TestCase, main, mock

import numpy as np

from learning.task_disagreement_trace_v1.test_extract import fixture
from learning.task_student_collection_v1.test_sampling import TASK, CROP, trace
from learning.task_state_residual_v1 import core as state
from learning.task_pair_fresh_multi_v1 import entry as owner_entry
from . import diagnostic, experiment


class FakeGame:
    task_sha256, crop_sha256 = TASK, CROP
    identity = dict(source_role='A05',batch=2,role='T01',opponent='market',side=0)
    def __init__(self, raw, replay): self.raw, self.replay, self.closed = raw, replay, False
    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    @contextmanager
    def open_trace(self, *, guard):
        yield io.BytesIO(self.raw)
        self.trace_verified = True


def fake_case():
    args = fixture()
    args['role'] = 'A05'
    obs = args['inputs']['observation']
    if obs['player'] != 0:
        raise AssertionError('fixture must represent student seat 0')
    frame = [{'observation':obs}, {'observation':obs}]
    replay = {'steps':[frame] * 720, 'configuration':args['inputs']['configuration']}
    record = dict(source='callback',inputs=args['inputs'],
                  teacher_label=args['teacher_label'],chosen=args['chosen'])
    game = FakeGame(trace({args['step']:[record]}), replay)
    return args, game


class IntegrationTests(TestCase):
    def test_real_full_scanner_view_and_extract_are_connected(self):
        args, game = fake_case()
        def reader(_context, **_kwargs): return game
        records = experiment._collect_case(None, ('A05','T01','market',0), args['metadata'],
                                           lambda:None, read_game=reader)
        self.assertTrue(game.trace_verified and game.closed)
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0].role,records[0].opponent,records[0].side),('A05','market',0))
        self.assertEqual(records[0].student_index, 1)

    def test_seat_one_uses_only_its_own_private_replay_view(self):
        args = fixture(); args['role'] = 'A06'
        inputs = args['inputs']; inputs['observation']['player'] = 1
        own = deepcopy(inputs['observation'])
        other = deepcopy(own); other['player'] = 0
        other['private']['shed']['WHEAT'] = 7
        self.assertNotEqual(other['private'], own['private'])
        frame = [{'observation':other}, {'observation':own}]
        replay = {'steps':[frame] * 720, 'configuration':inputs['configuration']}
        record = dict(source='callback', inputs=inputs,
                      teacher_label=args['teacher_label'], chosen=args['chosen'])
        game = FakeGame(trace({args['step']:[record]}), replay)
        game.identity = dict(source_role='A06',batch=2,role='T02',opponent='round2',side=1)
        records = experiment._collect_case(None,('A06','T02','round2',1),args['metadata'],
                                            lambda:None,read_game=lambda *_a,**_k:game)
        self.assertEqual(len(records),1)
        self.assertEqual((records[0].role,records[0].side),('A06',1))
        self.assertTrue(game.trace_verified and game.closed)
        own['private']['shed']['WHEAT'] = 8
        game = FakeGame(trace({args['step']:[record]}), replay)
        game.identity = dict(source_role='A06',batch=2,role='T02',opponent='round2',side=1)
        with self.assertRaises(ValueError):
            experiment._collect_case(None,('A06','T02','round2',1),args['metadata'],
                                     lambda:None,read_game=lambda *_a,**_k:game)

    def test_replay_view_private_order_or_scalar_type_difference_stops(self):
        for change in ('private','config','scalar'):
            args, game = fake_case()
            game.replay = deepcopy(game.replay)
            frame = game.replay['steps'][args['step']]
            if change == 'private':
                shed = frame[0]['observation']['private']['shed']
                shed['WHEAT'] = shed.get('WHEAT', 0) + 1
            elif change == 'config':
                game.replay['configuration']['boardSize'] += 1
            else:
                frame[0]['observation']['step'] = float(frame[0]['observation']['step'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                experiment._collect_case(None,('A05','T01','market',0),args['metadata'],
                    lambda:None, read_game=lambda *_a,**_k:game)
            self.assertTrue(game.closed)

    def test_true_assessment_report_and_tampering(self):
        args, game = fake_case()
        records = experiment._collect_case(None,('A05','T01','market',0),args['metadata'],
                                             lambda:None,read_game=lambda *_a,**_k:game)
        class Network:
            def score(self,_shared,candidates):
                return np.arange(len(candidates),dtype=np.float64)
        groups = diagnostic.score(records,Network(),guard=lambda:None)
        report = diagnostic.assess(records,groups,mean=np.zeros(373,np.float64),
            scale=np.ones(373,np.float64),params=state.initial_params(123),
            metadata=args['metadata'],guard=lambda:None)
        self.assertTrue(experiment._validate_report(report))
        damaged = deepcopy(report)
        damaged['by_role']['A05']['groups'] += 1
        with self.assertRaises(ValueError): experiment._validate_report(damaged)
        damaged = deepcopy(report)
        damaged['historical_screen'] = 'passed'
        with self.assertRaises(ValueError): experiment._validate_report(damaged)
        damaged = deepcopy(report)
        case = damaged['coverage_by_case']['A05']['market']['0']
        case['groups'],case['windows']['0'] = 5,5
        with self.assertRaises(ValueError): experiment._validate_report(damaged)
        damaged = deepcopy(report)
        case = damaged['coverage_by_case']['A05']['market']['0']
        case['groups'],case['windows']['0'] = 0,0
        with self.assertRaises(ValueError): experiment._validate_report(damaged)

    def test_owned_job_failure_cannot_write_completion(self):
        writes = {}
        owner = dict(schema='task-pair-fresh-multi-owner-v1',registration_commit='a'*40,
                     binding_sha256='b'*64,pid=1,created_us=1,deadline=1.0)
        with mock.patch.object(experiment.os,'name','nt'), \
             mock.patch.object(experiment.previous,'physical_available',return_value=8*1024**3), \
             mock.patch.object(experiment.shutil,'disk_usage',return_value=SimpleNamespace(free=8*1024**3)), \
             mock.patch.object(experiment,'_runtime_binding',return_value=('a'*40,{},'b'*64)), \
             mock.patch.object(experiment.previous,'_write_new',side_effect=lambda p,v:writes.setdefault(p.name,v)), \
             mock.patch.object(owner_entry,'owner_marker',return_value=experiment.previous._encoded(owner)), \
             mock.patch('psutil.Process',return_value=SimpleNamespace(create_time=lambda:1)), \
             mock.patch.object(experiment.supervisor,'_supervise_owned',return_value={}), \
             mock.patch.object(experiment.transfer,'_job_ok',return_value=False):
            with self.assertRaises(RuntimeError): experiment.run()
        self.assertIn('failed.local.json',writes)
        self.assertNotIn('closed.local.json',writes)
        self.assertIs(writes['failed.local.json']['retry_allowed'],False)


if __name__ == '__main__': main()
