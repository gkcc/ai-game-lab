"""In-memory checks for owned cases, seat selection and sealed aggregates."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import experiment as e


def aggregate():
    return dict(teacher_callbacks=1, terminal_callbacks=1, teacher_empty_turn=0, terminal_empty_turn=0,
        supported=1, unsupported=0, unknown=0, ambiguous=0,
        supported_target_features_equal=1, supported_target_features_differ=0)


def case(head, opponent, side):
    rows = [dict(step=step, teacher_request_matched=True, terminal_request_equal=True, **aggregate())
            for step in e.STEPS]
    return dict(schema='teacher-support-probe-private-case-v1', registration_commit=head, role='A01',
        opponent=opponent, learner_side=side, teacher_side=1-side, replay_states=720,
        sampled_turns=6, rule_calls=12, rows=rows,
        totals={key: sum(row[key] for row in rows) for key in e.COUNT_FIELDS},
        official_transitions=0, training=False, passed=True)


def successful_job():
    return dict(passed=True, launcher_reaped=True, job_members_zero=True, exit_code=0,
                retry_allowed=False, stop_reason=None, error_kind=None, cleanup_error_kind=None,
                rss_complete_samples=1)


class RowTests(unittest.TestCase):
    def test_support_counts_and_empty_turns_are_consistent(self):
        row = case('a' * 40, 'market', 0)['rows'][0]
        e._validate_row(row, e.STEPS[0])
        for key, value in (('unknown', 1), ('ambiguous', 1), ('teacher_empty_turn', 1),
                           ('terminal_request_equal', 1), ('supported_target_features_equal', 0)):
            changed = dict(row, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                e._validate_row(changed, e.STEPS[0])
        impossible = dict(row, terminal_callbacks=0, terminal_empty_turn=1)
        with self.assertRaises(ValueError): e._validate_row(impossible, e.STEPS[0])
        with self.assertRaises(ValueError): e._validate_row(dict(row, raw_observation={}), e.STEPS[0])
        invalid_case = case('a' * 40, 'market', 0)
        invalid_case['raw_observation'] = {}
        with self.assertRaises(ValueError):
            e._case_result(e.previous._encoded(invalid_case), head='a' * 40, opponent='market', side=0)


class ParentTests(unittest.TestCase):
    def setUp(self):
        self.head, self.files, self.calls = 'a' * 40, {}, []
        self.files['binding.local.json'] = e.previous._encoded({'synthetic': True})

    def write(self, path, value):
        name = Path(path).name
        if name in self.files: raise FileExistsError(name)
        self.files[name] = e.previous._encoded(value)

    def read(self, path):
        name = Path(path).name
        if name not in self.files: raise FileNotFoundError(name)
        return self.files[name]

    def supervise(self, command, cwd, limits, job_name):
        opponent, side = command[-2], int(command[-1])
        name = e.replay_probe._case_name(opponent, side)
        self.calls.append(name)
        self.assertIn('claim-' + name + '.local.json', self.files)
        self.write(Path('child-attempt-' + name + '.local.json'), dict(
            schema='teacher-support-probe-child-attempt-v1', commit=self.head, case=name, retry_allowed=False))
        self.write(Path('case-' + name + '.local.json'), case(self.head, opponent, side))
        return successful_job()

    def run_with(self, fn):
        with ExitStack() as stack:
            for p in (
                patch.object(e, 'PRIVATE', Path('synthetic-private')),
                patch.object(e, '_runtime_binding', return_value=(self.head, {})),
                patch.object(e.previous, 'physical_available', return_value=8 * 1024**3),
                patch.object(e.previous, '_guard', return_value=None),
                patch.object(e.previous, '_write_new', side_effect=self.write),
                patch.object(e.previous, '_read', side_effect=self.read),
                patch.object(e.shutil, 'disk_usage', return_value=SimpleNamespace(free=16 * 1024**3)),
                patch.object(e.replay_probe, '_supervise_owned', side_effect=self.supervise),
                patch.object(Path, 'exists', new=lambda path: path.name in self.files),
                patch.dict(e.os.environ, {}, clear=False),
            ): stack.enter_context(p)
            with redirect_stdout(io.StringIO()): return fn()

    def test_all_cases_seal_and_tampering_or_failure_is_rejected(self):
        def exercise():
            e.run()
            result = e.load_completed()
            self.assertEqual(len(self.calls), 4)
            self.assertEqual(result['summary']['sampled_turns'], 24)
            self.assertEqual(result['summary']['totals']['supported'], 24)
            key = 'case-market-learner0.local.json'
            original = self.files[key]
            self.files[key] += b' '
            with self.assertRaises(ValueError): e.load_completed()
            self.files[key] = original
            self.files['failed.local.json'] = b'{}'
            with self.assertRaises(ValueError): e.load_completed()
        self.run_with(exercise)

    def test_failure_stops_before_later_cases(self):
        original = self.supervise
        def fail_second(command, cwd, limits, name):
            if len(self.calls) == 1:
                self.calls.append('failure')
                return {**successful_job(), 'passed': False, 'stop_reason': 'available_memory_limit'}
            return original(command, cwd, limits, name)
        self.supervise = fail_second
        def exercise():
            with self.assertRaises(RuntimeError): e.run()
            self.assertEqual(len(self.calls), 2)
            self.assertIn('failed.local.json', self.files)
            self.assertNotIn('closed.local.json', self.files)
        self.run_with(exercise)

    def test_parent_deadline_reserves_two_cleanup_intervals(self):
        def exercise():
            with patch.object(e.time, 'monotonic', side_effect=[0, 1000, 2000, 3000, 3580]):
                with self.assertRaises(TimeoutError): e.run()
            self.assertEqual(len(self.calls), 3)
            self.assertIn('failed.local.json', self.files)
        self.run_with(exercise)


class ChildTests(unittest.TestCase):
    def test_failed_parent_cannot_read_a_source(self):
        with patch.object(e, 'PRIVATE', Path('synthetic-private')), \
                patch.object(e, '_runtime_binding', return_value=('a' * 40, {})), \
                patch.object(Path, 'exists', new=lambda path: path.name == 'failed.local.json'), \
                patch.object(e.previous, '_read', side_effect=AssertionError('unexpected read')):
            with self.assertRaises(ValueError): e.child('market', 0)

    def test_both_teacher_seats_and_mismatch_stop(self):
        import decision.api as decision_api
        import decision.audit_market_teacher as audit
        import learning.task_student_anchor_collection_v1.sources as sources
        import learning.task_teacher_dev_deviation_v1.membership as membership
        import learning.task_pair_fresh_multi_v1.entry as owner
        from . import opponent, terminal, alignment
        head = 'a' * 40
        action = dict(farmer=['PASS'], hands=[], market=[])

        def exercise(side, mismatch):
            name, teacher_side = 'market-learner' + str(side), 1-side
            binding_raw = e.previous._encoded({'synthetic': True})
            files = {'binding.local.json': binding_raw,
                'claim-' + name + '.local.json': e.previous._encoded(dict(
                    schema='teacher-support-probe-claim-v1', commit=head, case=name, attempt=1,
                    retry_allowed=False, freeze_sha256=e.previous._sha(binding_raw), job_token_sha256='b'*64)),
                'attempt.local.json': e.previous._encoded(dict(schema='teacher-support-probe-attempt-v1',
                                                               commit=head, retry_allowed=False)),
                'owner.local.json': e.previous._encoded({'synthetic': True})}
            frames = [[dict(action=deepcopy(action), observation=dict(player=p, step=step,
                         private=dict(own=p))) for p in (0, 1)] for step in range(720)]
            if mismatch: frames[e.STEPS[0]+1][teacher_side]['action']['market'] = [['SELL', 'WHEAT', 1]]
            seen, cleared = [], []

            class Context:
                task_sha256, crop_sha256 = 'x', 'y'
                def source_role(self, **unused): return 'A01'
                def private_role_binding(self): return {'A01': 9}
                def model_normalization(self):
                    return None, None, dict(schema_version='s', schema_sha256='h', feature_names=[], shared_feature_names=[])
            class Game:
                def __enter__(self): return self
                def __exit__(self, *unused): return False
                @property
                def replay(self): return dict(steps=frames, configuration={})
            class Probe:
                def __init__(self, *args, **kwargs): self.last = None
                def __call__(self, obs, cfg): self.last = {'records': [{}]}; return deepcopy(action)
                def clear(self): self.last = None; cleared.append(True)
            def read(path): return files[Path(path).name]
            def write(path, value):
                key = Path(path).name
                if key in files: raise FileExistsError(key)
                files[key] = e.previous._encoded(value)
            def visible(frame, player):
                seen.append(player)
                return deepcopy(frame[player]['observation'])
            binding = dict(source_seed=9, task_sha256='x', crop_sha256='y')
            with ExitStack() as stack:
                for p in (
                    patch.object(e, 'PRIVATE', Path('synthetic-private')),
                    patch.object(e, '_runtime_binding', return_value=(head, binding)),
                    patch.object(Path, 'exists', new=lambda path: path.name in files),
                    patch.object(e.previous, '_read', side_effect=read),
                    patch.object(e.previous, '_write_new', side_effect=write),
                    patch.object(e.previous, '_guard', return_value=None),
                    patch.object(membership, 'verify_worker', return_value=None),
                    patch.object(owner, 'verify_live_owner', return_value=True),
                    patch.object(sources, 'open_context', return_value=Context()),
                    patch.object(sources, 'read_game', return_value=Game()),
                    patch.object(e.replay_probe, '_rival_source', return_value=(b'synthetic', 'f'*64)),
                    patch.object(opponent, 'SupportTeacher', Probe),
                    patch.object(terminal, 'TerminalTeacherProbe', Probe),
                    patch.object(alignment, 'align_turn', return_value=aggregate()),
                    patch.object(audit, 'observation', side_effect=visible),
                    patch.object(decision_api, 'observe', side_effect=lambda obs, cfg:
                                 SimpleNamespace(observation=obs, configuration=cfg)),
                    patch.dict(e.os.environ, {}, clear=False),
                ): stack.enter_context(p)
                if mismatch:
                    with self.assertRaises(ValueError): e.child('market', side)
                    failed = e.previous._document(files['case-' + name + '.failed.local.json'])
                    self.assertEqual((failed['completed_turns'], failed['rule_calls_started']), (0, 1))
                else:
                    e.child('market', side)
                    result = e._case_result(files['case-' + name + '.local.json'],
                                            head=head, opponent='market', side=side)
                    self.assertEqual((result['sampled_turns'], result['rule_calls'], result['teacher_side']),
                                     (6, 12, teacher_side))
                    self.assertEqual(len(cleared), 12)
            self.assertTrue(seen and set(seen) == {teacher_side})

        for side in (0, 1):
            exercise(side, False)
            exercise(side, True)


class CliTests(unittest.TestCase):
    def test_bad_arguments_do_not_echo_private_text(self):
        secret = 'synthetic_private_location'
        output = io.StringIO()
        with patch.object(sys, 'argv', ['experiment', 'child', secret, '0']), redirect_stderr(output):
            with self.assertRaises(SystemExit) as caught: e.cli()
        self.assertEqual(caught.exception.code, 1)
        self.assertNotIn(secret, output.getvalue())


if __name__ == '__main__':
    unittest.main()
