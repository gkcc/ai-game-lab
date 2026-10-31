"""In-memory child and parent control-flow checks; no real game or model.

The fixture's replays, observations, rules, jobs and private files are
synthetic. These checks do not establish official replay or GPU behavior.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import psutil

import decision.api as decision_api
import decision.audit_market_teacher as audit
import learning.task_pair_fresh_multi_v1.entry as owner
import learning.task_student_anchor_collection_v1.sources as sources
import learning.task_teacher_dev_deviation_v1.membership as membership
import learning.task_teacher_support_probe_v1.opponent as opponent_probe
import learning.task_teacher_support_probe_v1.terminal as terminal_probe

from . import dataset, experiment as e, extract
from .test_dataset import matrix


def success_job():
    return dict(passed=True, assigned=True, resumed=True, launcher_reaped=True, job_members_zero=True, exit_code=0,
                retry_allowed=False, stop_reason=None, error_kind=None, cleanup_error_kind=None,
                acceptance_error_kind=None, termination_error_kind=None, wait_error_kind=None,
                membership_error_kind=None, close_error_kind=None, rss_complete_samples=1,
                peak_sampled_rss_bytes=1, peak_sampled_job_commit_bytes=1, max_sampled_members=1,
                rss_threshold_sample_members=None)


def support_counts(total=5, supported=1):
    return dict(teacher_callbacks=total, terminal_callbacks=total,
                teacher_empty_turn=int(total == 0), terminal_empty_turn=int(total == 0),
                supported=supported, unsupported=0, unknown=total - supported, ambiguous=0,
                supported_target_features_equal=supported, supported_target_features_differ=0)


class MemoryFiles:
    """Exclusive synthetic files keyed by basename, never touching the disk."""

    def __init__(self):
        self.data = {}

    def read(self, path):
        name = Path(path).name
        if name not in self.data:
            raise FileNotFoundError('synthetic file missing')
        return self.data[name]

    def write(self, path, value):
        name = Path(path).name
        if name in self.data:
            raise FileExistsError('synthetic file already exists')
        self.data[name] = e.previous._encoded(value)

    def open(self, path, mode='r', *args, **kwargs):
        name = Path(path).name
        if mode == 'rb':
            return io.BytesIO(self.read(path))
        if mode == 'xb':
            if name in self.data:
                raise FileExistsError('synthetic file already exists')
            outer = self

            class Writer(io.BytesIO):
                def __exit__(self, exc_type, exc, tb):
                    if exc_type is None:
                        outer.data[name] = self.getvalue()
                    return super().__exit__(exc_type, exc, tb)

            return Writer()
        raise AssertionError('Unexpected synthetic file mode')


class ChildIntegrationTests(unittest.TestCase):
    """Run the real child orchestration over synthetic completed-game stubs."""

    def exercise(self, side, mode):
        role, rival, head = 'A01', 'market', 'a' * 40
        name = e._case_name(role, rival, side)
        teacher_side = 1 - side
        files = MemoryFiles()
        binding = dict(source_roles={role: 9}, task_sha256='x', crop_sha256='y')
        files.write('binding.local.json', {'synthetic': True})
        binding_sha = e.previous._sha(files.read('binding.local.json'))
        files.write('claim-' + name + '.local.json', dict(
            schema='teacher-transfer-dataset-claim-v1', commit=head, case=name, attempt=1,
            retry_allowed=False, freeze_sha256=binding_sha, job_token_sha256='b' * 64))
        files.write('attempt.local.json', dict(schema='teacher-transfer-dataset-attempt-v1',
                                               commit=head, retry_allowed=False))
        files.write('owner.local.json', {'synthetic': True})
        action = dict(farmer=['PASS'], hands=[], market=[])
        frames = [[dict(action=deepcopy(action), observation=dict(
            player=player, step=step, hour=step % 24, private=dict(owner=player)))
            for player in (0, 1)] for step in range(720)]
        replay = dict(steps=frames, configuration={'turnsPerDay': 24})
        observed_sides, events, cleared, saved = [], [], [], []
        true_rank = dataset.rank_ordinals

        class Context:
            task_sha256, crop_sha256 = 'x', 'y'
            def source_role(self, **unused): return role
            def private_role_binding(self): return {role: 9}
            def model_normalization(self): return None, None, {}

        class Game:
            def __enter__(self): return self
            def __exit__(self, *unused): return False
            @property
            def replay(self): return replay

        class Probe:
            def __init__(self, *args, **kwargs): self.last = None
            def __call__(self, observation, configuration):
                self.last = {'records': [{'synthetic_ordinal': i} for i in range(5)]}
                return deepcopy(action)
            def clear(self):
                self.last = None
                cleared.append(True)

        def visible(frame, player):
            observed_sides.append(player)
            return deepcopy(frame[player]['observation'])

        def rank(r, o, s, step, total):
            events.append(('rank', step))
            return true_rank(r, o, s, step, total)

        def extraction(obs, cfg, teacher_records, terminal_records, *, schema, guard):
            step = obs['step']
            events.append(('extract', step))
            if mode == 'fail' and step == e.STEPS[1]:
                raise MemoryError('synthetic failure')
            selected = true_rank(role, rival, side, step, len(teacher_records))
            excluded = next(index for index in range(5) if index not in selected)
            ordinal = selected[0] if mode == 'positive' else excluded
            item = SimpleNamespace(candidate_count=3, features=matrix(step), label=0,
                                   operation='WATER', teacher_ordinal=ordinal, terminal_ordinal=ordinal)
            return (item,), support_counts()

        def save_case(_path, groups, guard):
            guard()
            saved.extend((step, item.teacher_ordinal) for step, item in groups)
            return 'c' * 64

        with ExitStack() as stack:
            for p in (
                patch.object(e, 'PRIVATE', Path('synthetic-private')),
                patch.object(e, '_runtime_binding', return_value=(head, binding)),
                patch.object(e, '_registered_binding', return_value=binding_sha),
                patch.object(Path, 'exists', new=lambda path: path.name in files.data),
                patch.object(e.previous, '_read', side_effect=files.read),
                patch.object(e.previous, '_write_new', side_effect=files.write),
                patch.object(e.previous, '_guard', return_value=None),
                patch.object(e, '_model_schema', return_value={}),
                patch.object(e, '_save_case', side_effect=save_case),
                patch.object(membership, 'verify_worker', return_value=None),
                patch.object(owner, 'verify_live_owner', return_value=True),
                patch.object(sources, 'open_context', return_value=Context()),
                patch.object(sources, 'read_game', return_value=Game()),
                patch.object(e.replay_probe, '_rival_source', return_value=(b'synthetic', 'f' * 64)),
                patch.object(opponent_probe, 'SupportTeacher', Probe),
                patch.object(terminal_probe, 'TerminalTeacherProbe', Probe),
                patch.object(dataset, 'rank_ordinals', side_effect=rank),
                patch.object(extract, 'extract_turn', side_effect=extraction),
                patch.object(audit, 'observation', side_effect=visible),
                patch.object(decision_api, 'observe', side_effect=lambda obs, cfg:
                             SimpleNamespace(observation=obs, configuration=cfg)),
                patch.dict(e.os.environ, {}, clear=False),
            ):
                stack.enter_context(p)
            if mode == 'fail':
                with self.assertRaises(MemoryError):
                    e.child(role, rival, side)
                failed = e.previous._document(files.read('case-' + name + '.failed.local.json'))
                self.assertEqual((failed['completed_turns'], failed['rule_calls_started']), (1, 4))
                self.assertNotIn('case-' + name + '.local.json', files.data)
                self.assertFalse(saved)
                # The earlier successful turn released both probe records. The
                # failed turn exits its child process; this in-memory test does
                # not pretend to verify operating-system Job cleanup.
                self.assertEqual(len(cleared), 2)
            else:
                e.child(role, rival, side)
                case = e._case_result(files.read('case-' + name + '.local.json'), head=head,
                                      role=role, opponent=rival, side=side)
                self.assertEqual((case['sampled_turns'], case['rule_calls'], case['teacher_side']),
                                 (6, 12, teacher_side))
                self.assertEqual([row['step'] for row in case['rows']], list(e.STEPS))
                self.assertEqual([row['hour'] for row in case['rows']], list(e.HOURS))
                self.assertEqual(len(cleared), 12)
                if mode == 'excluded':
                    self.assertEqual(saved, [])
                    self.assertTrue(all(row['supported'] == 1 and row['retained_groups'] == 0
                                        for row in case['rows']))
                else:
                    self.assertEqual(saved, [(step, true_rank(role, rival, side, step, 5)[0])
                                             for step in e.STEPS])
                    self.assertTrue(all(row['retained_groups'] == 1 for row in case['rows']))
            self.assertTrue(observed_sides and set(observed_sides) == {teacher_side})
            for step in {step for kind, step in events if kind == 'extract'}:
                self.assertLess(events.index(('rank', step)), events.index(('extract', step)))

    def test_unselected_supported_callback_is_not_backfilled_on_either_seat(self):
        # At this fixed coordinate the unselected callback is specifically the
        # fifth one (ordinal 4); the other coordinates cover arbitrary omissions.
        self.assertNotIn(4, dataset.rank_ordinals('A01', 'market', 0, e.STEPS[0], 5))
        for side in (0, 1):
            with self.subTest(side=side):
                self.exercise(side, 'excluded')

    def test_selected_supported_callback_is_retained_on_either_seat(self):
        for side in (0, 1):
            with self.subTest(side=side):
                self.exercise(side, 'positive')

    def test_source_failure_stops_without_partial_case_output(self):
        self.exercise(0, 'fail')


class ParentIntegrationTests(unittest.TestCase):
    """Run the real 16-case coordinator against exclusively synthetic jobs."""

    def setUp(self):
        self.files, self.head, self.calls = MemoryFiles(), 'a' * 40, []
        self.binding = dict(source_roles={'A01': 1, 'A02': 2, 'A07': 3, 'A08': 4},
                            task_sha256='x', crop_sha256='y')
        self.files.write('binding.local.json', self.binding)
        self.binding_sha = e.previous._sha(self.files.read('binding.local.json'))
        self.registration = e.previous._encoded({**e._registration(),
                                                'binding_commitment_sha256': self.binding_sha})
        self.fail_case = None

    def git(self, command, **kwargs):
        if command[-2] == 'show':
            return self.registration
        if command[-2] == 'rev-parse':
            return (e.BASE_COMMIT + '\n').encode('ascii')
        raise AssertionError('Unexpected synthetic Git request')

    def supervise(self, command, cwd, limits, job_name):
        role, rival, side = command[-3], command[-2], int(command[-1])
        name = e._case_name(role, rival, side)
        self.calls.append(name)
        self.assertIn('claim-' + name + '.local.json', self.files.data)
        if self.fail_case == len(self.calls):
            return {**success_job(), 'passed': False, 'stop_reason': 'available_memory_limit'}
        self.files.write('child-attempt-' + name + '.local.json', dict(
            schema='teacher-transfer-dataset-child-attempt-v1', commit=self.head,
            case=name, retry_allowed=False))
        rows, groups = [], []
        for index, (step, hour) in enumerate(zip(e.STEPS, e.HOURS)):
            selected = dataset.rank_ordinals(role, rival, side, step, 5)
            ordinal = selected[0]
            operation = 'DIG' if index % 2 else 'WATER'
            # Every synthetic group has distinct features; this checks the
            # entire parent NPZ -> merge -> split -> sealed reload path.
            value = len(self.calls) * 100 + index
            item = SimpleNamespace(candidate_count=3, features=matrix(value, op=operation),
                                   label=0, operation=operation,
                                   teacher_ordinal=ordinal, terminal_ordinal=ordinal)
            groups.append((step, item))
            rows.append(dict(step=step, hour=hour, teacher_request_matched=True,
                             terminal_request_equal=True, selected_ordinals=list(selected),
                             retained_groups=1, **support_counts()))
        data_path = e.PRIVATE / ('data-' + name + '.local.npz')
        data_sha = e._save_case(data_path, groups, lambda: None)
        self.files.write('case-' + name + '.local.json', dict(
            schema='teacher-transfer-dataset-private-case-v1', registration_commit=self.head, role=role,
            opponent=rival, learner_side=side, teacher_side=1-side, replay_states=720,
            sampled_turns=6, rule_calls=12, rows=rows, data_sha256=data_sha,
            official_transitions=0, training=False, passed=True))
        return success_job()

    def run_with(self, operation):
        fixture = self
        class Context:
            task_sha256, crop_sha256 = 'x', 'y'
            def private_role_binding(self): return dict(fixture.binding['source_roles'])
            def source_role(self, *, batch, role):
                return next(key for key, value in e.SLOTS.items() if value == role)
            def model_normalization(self): return np.zeros(373), np.ones(373), {}

        with ExitStack() as stack:
            for p in (
                patch.object(e, 'PRIVATE', Path('synthetic-private')),
                patch.object(e, '_runtime_binding', return_value=(self.head, self.binding)),
                patch.object(e, '_model_schema', return_value={}),
                patch.object(e.previous, 'physical_available', return_value=8 * 1024**3),
                patch.object(e.previous, '_guard', return_value=None),
                patch.object(e.previous, '_write_new', side_effect=self.files.write),
                patch.object(e.previous, '_read', side_effect=self.files.read),
                patch.object(Path, 'exists', new=lambda path: path.name in self.files.data),
                patch.object(Path, 'open', new=lambda path, mode='r', *a, **kw:
                             self.files.open(path, mode, *a, **kw)),
                patch.object(e.shutil, 'disk_usage', return_value=SimpleNamespace(free=16 * 1024**3)),
                patch.object(e.replay_probe, '_supervise_owned', side_effect=self.supervise),
                patch.object(e.subprocess, 'check_output', side_effect=self.git),
                patch.object(sources, 'open_context', return_value=Context()),
                patch.object(psutil, 'Process', return_value=SimpleNamespace(create_time=lambda: 1.0)),
                patch.dict(e.os.environ, {}, clear=False),
            ):
                stack.enter_context(p)
            with redirect_stdout(io.StringIO()):
                return operation()

    def test_sixteen_jobs_merge_seal_reload_and_reject_tampering(self):
        def exercise():
            e.run()
            result, contract = e.load_completed()
            self.assertEqual(len(self.calls), 16)
            summary = result['summary']
            self.assertEqual((summary['source_games'], summary['sampled_turns'], summary['rule_calls']),
                             (16, 96, 192))
            self.assertEqual((summary['source_groups'], summary['train_groups'], summary['dev_groups']),
                             (96, 48, 48))
            self.assertEqual(summary['status'], 'passed')
            self.assertFalse(summary['training'])
            self.assertFalse(summary['independent_evaluation'])
            with np.load(io.BytesIO(self.files.read('train.local.npz')), allow_pickle=False) as train:
                self.assertEqual(set(train['role'].tolist()), {0, 1})
            with np.load(io.BytesIO(self.files.read('dev.local.npz')), allow_pickle=False) as dev:
                self.assertEqual(set(dev['role'].tolist()), {2, 3})
            self.assertFalse(contract['training_started'])
            train_groups, dev_groups, returned_contract = e.load_trainable()
            self.assertEqual((len(train_groups), len(dev_groups)), (48, 48))
            self.assertEqual(returned_contract, contract)
            reads = []
            def racing_binding_read(path):
                raw = self.files.read(path)
                if Path(path).name == 'binding.local.json':
                    reads.append(True)
                    if len(reads) == 2:
                        return e.previous._encoded({**self.binding, 'task_sha256': 'changed'})
                return raw
            with patch.object(e.previous, '_read', side_effect=racing_binding_read):
                with self.assertRaises(ValueError): e.load_trainable()
            self.assertGreaterEqual(len(reads), 2)
            for name in ('claim-' + self.calls[0] + '.local.json',
                         'child-attempt-' + self.calls[0] + '.local.json', 'binding.local.json'):
                original = self.files.data[name]
                self.files.data[name] = original + b' '
                with self.subTest(name=name), self.assertRaises(ValueError):
                    e.load_completed()
                self.files.data[name] = original
            # A consistent local reseal can fool hash-only reporting. It must
            # not authorize training if its gate does not rederive from cases.
            original_files = dict(self.files.data)
            result = e.previous._document(self.files.read('result.provisional.local.json'))
            result['summary']['train_groups'] += 1
            self.files.data['result.provisional.local.json'] = e.previous._encoded(result)
            closed = e.previous._document(self.files.read('closed.local.json'))
            closed['result_sha256'] = e.previous._sha(self.files.read('result.provisional.local.json'))
            self.files.data['closed.local.json'] = e.previous._encoded(closed)
            e.load_completed()
            with self.assertRaises(ValueError): e.load_trainable()
            self.files.data = dict(original_files)

            # A resealed split with an altered label also passes the technical
            # hash check but must differ from the rederived case groups.
            with np.load(io.BytesIO(self.files.read('train.local.npz')), allow_pickle=False) as data:
                altered = {key: data[key].copy() for key in data.files}
            altered['target'][0] = (altered['target'][0] + 1) % 3
            output = io.BytesIO()
            np.savez_compressed(output, **altered)
            self.files.data['train.local.npz'] = output.getvalue()
            changed_contract = e.previous._document(self.files.read('dataset-contract.local.json'))
            changed_contract['train_sha256'] = e.previous._sha(output.getvalue())
            self.files.data['dataset-contract.local.json'] = e.previous._encoded(changed_contract)
            closed = e.previous._document(self.files.read('closed.local.json'))
            closed['dataset_contract_sha256'] = e.previous._sha(self.files.read('dataset-contract.local.json'))
            self.files.data['closed.local.json'] = e.previous._encoded(closed)
            e.load_completed()
            with self.assertRaises(ValueError): e.load_trainable()
            self.files.data = original_files
        self.run_with(exercise)

    def test_reject_internally_inconsistent_job_receipts(self):
        good = success_job()
        self.assertTrue(e._job_ok(good))
        for key, bad in (('assigned', False), ('resumed', False), ('exit_code', False),
                         ('acceptance_error_kind', 'NoCompleteSample'), ('wait_error_kind', 'WaitError'),
                         ('peak_sampled_rss_bytes', e.JOB['max_rss_bytes'] + 1),
                         ('peak_sampled_job_commit_bytes', e.JOB['max_commit_bytes'] + 1),
                         ('rss_complete_samples', 0)):
            with self.subTest(key=key):
                self.assertFalse(e._job_ok({**good, key: bad}))

    def test_failed_data_gate_cannot_enter_training(self):
        with patch.object(e, 'load_completed', return_value=({'summary': {'status': 'failed'}}, {})), \
                patch.object(sources, 'open_context', side_effect=AssertionError('must not read source')):
            with self.assertRaises(ValueError): e.load_trainable()

    def test_failed_job_stops_and_attempt_cannot_be_retried(self):
        self.fail_case = 2
        def exercise():
            with self.assertRaises(RuntimeError):
                e.run()
            self.assertEqual(len(self.calls), 2)
            self.assertIn('failed.local.json', self.files.data)
            self.assertNotIn('closed.local.json', self.files.data)
            failed_job = e.previous._document(self.files.read('job-' + self.calls[-1] + '.local.json'))
            self.assertTrue(failed_job['launcher_reaped'] and failed_job['job_members_zero'])
            with self.assertRaises(ValueError):
                e.load_completed()
            with self.assertRaises(FileExistsError):
                e.run()
            self.assertEqual(len(self.calls), 2)
        self.run_with(exercise)


if __name__ == '__main__':
    unittest.main()
