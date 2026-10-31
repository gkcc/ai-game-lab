"""Focused in-memory checks for teacher case inputs and owned-job sealing."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_student_collection_v1.test_dataset import inputs
from . import experiment as e


def case(head, opponent, side):
    return dict(schema='multi-teacher-probe-private-case-v1', registration_commit=head, role='A01',
        opponent=opponent, learner_side=side, teacher_side=1-side, replay_states=720, turns=719,
        requests_matched=719, callbacks=1, max_candidate_width=3,
        operations=dict(DIG=1, FERTILIZE=0, HARVEST=0, PLANT=0, WATER=0),
        official_transitions=0, training=False, passed=True)


def successful_job():
    return dict(passed=True, launcher_reaped=True, job_members_zero=True, exit_code=0,
                retry_allowed=False, stop_reason=None, error_kind=None, cleanup_error_kind=None,
                rss_complete_samples=1)


class RecordTests(unittest.TestCase):
    def test_single_player_schema_labels_and_width_are_required(self):
        data = inputs()
        data['configuration'] = {}
        view = SimpleNamespace(observation=deepcopy(data['observation']), configuration={})
        schema = {key: data[key] for key in
                  ('schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names')}
        record = dict(inputs=data, teacher_label='id0', chosen='id0', op='DIG')
        self.assertEqual(e._check_record(record, view, schema), (3, 'DIG'))
        for mutate in ('different_view', 'unknown_label', 'duplicate_id', 'bad_width', 'nonfinite'):
            altered = deepcopy(record)
            if mutate == 'different_view': altered['inputs']['observation']['private'] = 'other player'
            if mutate == 'unknown_label': altered['teacher_label'] = altered['chosen'] = 'missing'
            if mutate == 'duplicate_id': altered['inputs']['candidates'][1]['id'] = 'id0'
            if mutate == 'bad_width': altered['inputs']['shared_x'].pop()
            if mutate == 'nonfinite': altered['inputs']['candidates'][0]['x'][0] = float('nan')
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                e._check_record(altered, view, schema)


class HistoricalFailureTests(unittest.TestCase):
    def test_old_failure_requires_its_unique_attempt(self):
        commit = '7301c921cfd8ae8e5317cc9b4834165b50dc8dac'
        plan = e.previous._encoded({'synthetic': True})
        files = {'plan.local.json': plan,
                 'plan-frozen.local.json': e.previous._encoded(dict(registration_commit=commit,
                                                                  plan_sha256=e.previous._sha(plan))),
                 'failed.local.json': e.previous._encoded(dict(outcome='failed', error_type='MemoryError',
                    retry_allowed=False, comparisons_started=346, controller_calls_reserved=692,
                    official_transitions_reserved=692)),
                 'attempt.local.json': e.previous._encoded(dict(schema='student-anchor-mixed-attempt-v1',
                    commit=commit, max_comparisons=480, retry_allowed=False))}
        def read(path):
            name = Path(path).name
            if name not in files: raise FileNotFoundError(name)
            return files[name]
        with patch.object(e, 'FAILED_STAGE', Path('synthetic-private')), \
                patch.object(Path, 'exists', return_value=False), \
                patch.object(e.previous, '_read', side_effect=read):
            self.assertIn('attempt_sha256', e._failed_source_snapshot())
            original = files.pop('attempt.local.json')
            with self.assertRaises(FileNotFoundError): e._failed_source_snapshot()
            attempt = e.previous._document(original)
            attempt['retry_allowed'] = True
            files['attempt.local.json'] = e.previous._encoded(attempt)
            with self.assertRaises(ValueError): e._failed_source_snapshot()


class ParentTests(unittest.TestCase):
    def setUp(self):
        self.head, self.files, self.calls, self.limits = 'a' * 40, {}, [], []
        self.files['binding.local.json'] = e.previous._encoded({'synthetic': True})

    def write(self, path, value):
        name = Path(path).name
        if name in self.files:
            raise FileExistsError(name)
        self.files[name] = e.previous._encoded(value)

    def read(self, path):
        name = Path(path).name
        if name not in self.files:
            raise FileNotFoundError(name)
        return self.files[name]

    def supervise(self, command, cwd, limits, job_name):
        opponent, side = command[-2], int(command[-1])
        name = e._case_name(opponent, side)
        self.calls.append(name)
        self.limits.append(limits)
        self.assertTrue(job_name.startswith('Local\\TeacherDeviation-'))
        self.assertIn('claim-' + name + '.local.json', self.files)
        self.write(Path('child-attempt-' + name + '.local.json'), dict(
            schema='multi-teacher-probe-child-attempt-v1', commit=self.head, case=name, retry_allowed=False))
        self.write(Path('case-' + name + '.local.json'), case(self.head, opponent, side))
        return successful_job()

    def patches(self):
        return (patch.object(e, 'PRIVATE', Path('synthetic-private')),
                patch.object(e, '_runtime_binding', return_value=(self.head, {})),
                patch.object(e.previous, 'physical_available', return_value=8 * 1024**3),
                patch.object(e.previous, '_guard', return_value=None),
                patch.object(e.previous, '_write_new', side_effect=self.write),
                patch.object(e.previous, '_read', side_effect=self.read),
                patch.object(e.shutil, 'disk_usage', return_value=SimpleNamespace(free=16 * 1024**3)),
                patch.object(e, '_supervise_owned', side_effect=self.supervise),
                patch.object(Path, 'exists', new=lambda path: path.name in self.files),
                patch.dict(e.os.environ, {}, clear=False))

    def run_with(self, fn):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in self.patches(): stack.enter_context(item)
            with redirect_stdout(io.StringIO()):
                return fn()

    def test_four_cases_are_sealed_and_tampering_is_rejected(self):
        def exercise():
            e.run()
            self.assertEqual(len(self.calls), 4)
            result = e.load_completed()
            self.assertEqual(result['summary']['turns'], 2876)
            self.assertEqual(result['summary']['requests_matched'], 2876)
            name = 'case-market-learner0.local.json'
            original = self.files[name]
            self.files[name] += b' '
            with self.assertRaises(ValueError): e.load_completed()
            self.files[name] = original
            self.files['failed.local.json'] = b'{}'
            with self.assertRaises(ValueError): e.load_completed()
        self.run_with(exercise)

    def test_failed_job_stops_before_later_cases(self):
        original = self.supervise
        def stop_second(command, cwd, limits, job_name):
            if len(self.calls) == 1:
                self.calls.append('second-failed')
                return {**successful_job(), 'passed': False, 'stop_reason': 'available_memory_limit'}
            return original(command, cwd, limits, job_name)
        self.supervise = stop_second
        def exercise():
            with self.assertRaises(RuntimeError): e.run()
            self.assertEqual(len(self.calls), 2)
            self.assertNotIn('closed.local.json', self.files)
            self.assertIn('failed.local.json', self.files)
            with self.assertRaises(ValueError): e.load_completed()
        self.run_with(exercise)

    def test_parent_deadline_caps_each_job_and_stops_before_fourth(self):
        def exercise():
            with patch.object(e.time, 'monotonic', side_effect=[0, 1000, 2000, 3000, 3580]):
                with self.assertRaises(TimeoutError): e.run()
            self.assertEqual(len(self.calls), 3)
            self.assertEqual([item['wall_seconds'] for item in self.limits], [900, 900, 580])
            self.assertIn('failed.local.json', self.files)
            self.assertNotIn('closed.local.json', self.files)
        self.run_with(exercise)


class ChildEntryTests(unittest.TestCase):
    def test_child_never_reads_sources_after_parent_failure(self):
        with patch.object(e, 'PRIVATE', Path('synthetic-private')), \
                patch.object(e, '_runtime_binding', return_value=('a' * 40, {})), \
                patch.object(Path, 'exists', new=lambda path: path.name == 'failed.local.json'), \
                patch.object(e.previous, '_read', side_effect=AssertionError('unexpected read')):
            with self.assertRaises(ValueError): e.child('market', 0)

    def test_child_without_registered_job_cannot_start_source(self):
        import learning.task_teacher_dev_deviation_v1.membership as membership
        head, name = 'a' * 40, 'market-learner0'
        binding_raw = e.previous._encoded({'synthetic': True})
        binding_sha = e.previous._sha(binding_raw)
        claim = dict(schema='multi-teacher-probe-claim-v1', commit=head, case=name,
                     attempt=1, retry_allowed=False, freeze_sha256=binding_sha,
                     job_token_sha256='b' * 64)
        documents = {'binding.local.json': binding_raw,
            'claim-' + name + '.local.json': e.previous._encoded(claim),
            'attempt.local.json': e.previous._encoded(dict(schema='multi-teacher-probe-attempt-v1',
                                                          commit=head, retry_allowed=False)),
            'owner.local.json': e.previous._encoded({'synthetic': True})}
        def read(path): return documents[Path(path).name]
        with patch.object(e, 'PRIVATE', Path('synthetic-private')), \
                patch.object(e, '_runtime_binding', return_value=(head, {})), \
                patch.object(Path, 'exists', return_value=False), \
                patch.object(e.previous, '_read', side_effect=read), \
                patch.object(e.previous, '_write_new') as write, \
                patch.object(membership, 'verify_worker', side_effect=ValueError('not in job')):
            with self.assertRaises(ValueError): e.child('market', 0)
        write.assert_not_called()

    def test_child_replays_teacher_seat_and_stops_on_request_mismatch(self):
        from contextlib import ExitStack
        import decision.api as decision_api
        import decision.audit_market_teacher as audit
        import learning.task_student_anchor_collection_v1.sources as sources
        import learning.task_teacher_dev_deviation_v1.membership as membership
        import learning.task_pair_fresh_multi_v1.entry as owner
        from . import teacher

        head = 'a' * 40
        base = inputs()
        schema = {key: base[key] for key in
                  ('schema_version', 'schema_sha256', 'feature_names', 'shared_feature_names')}
        action = dict(farmer=['PASS'], hands=[], market=[])

        def exercise(mismatch, learner_side):
            name = 'market-learner' + str(learner_side)
            teacher_side = 1 - learner_side
            files = {}
            binding_raw = e.previous._encoded({'synthetic': True})
            binding_sha = e.previous._sha(binding_raw)
            files['binding.local.json'] = binding_raw
            files['claim-' + name + '.local.json'] = e.previous._encoded(dict(
                schema='multi-teacher-probe-claim-v1', commit=head, case=name, attempt=1,
                retry_allowed=False, freeze_sha256=binding_sha, job_token_sha256='b' * 64))
            files['attempt.local.json'] = e.previous._encoded(dict(
                schema='multi-teacher-probe-attempt-v1', commit=head, retry_allowed=False))
            files['owner.local.json'] = e.previous._encoded({'synthetic': True})
            frames = [[dict(action=deepcopy(action), observation=dict(player=side, step=step,
                         private=dict(own=side))) for side in (0, 1)] for step in range(720)]
            if mismatch:
                frames[6][teacher_side]['action']['market'] = [['SELL', 'WHEAT', 1]]
            replay = dict(steps=frames, configuration={})
            seen_sides, cleared = [], []

            class Context:
                task_sha256, crop_sha256 = 'x', 'y'
                def source_role(self, **unused): return 'A01'
                def private_role_binding(self): return {'A01': 9}
                def model_normalization(self): return None, None, schema

            class Game:
                def __enter__(self): return self
                def __exit__(self, *unused): return False
                @property
                def replay(self): return replay

            class Probe:
                def __init__(self, *args, **kwargs): self.last = None
                def __call__(self, obs, cfg):
                    data = inputs()
                    data['observation'], data['configuration'] = deepcopy(obs), deepcopy(cfg)
                    self.last = {'records': [dict(inputs=data, teacher_label='id0', chosen='id0', op='DIG')]}
                    return deepcopy(action)
                def clear(self):
                    self.last = None
                    cleared.append(True)

            def read(path): return files[Path(path).name]
            def write(path, value):
                key = Path(path).name
                if key in files: raise FileExistsError(key)
                files[key] = e.previous._encoded(value)
            def visible(frame, side):
                seen_sides.append(side)
                return deepcopy(frame[side]['observation'])
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
                    patch.object(e, '_rival_source', return_value=(b'synthetic', 'f' * 64)),
                    patch.object(teacher, 'TeacherProbe', Probe),
                    patch.object(audit, 'observation', side_effect=visible),
                    patch.object(decision_api, 'observe', side_effect=lambda obs, cfg:
                                 SimpleNamespace(observation=obs, configuration=cfg)),
                    patch.dict(e.os.environ, {}, clear=False),
                ): stack.enter_context(p)
                if mismatch:
                    with self.assertRaises(ValueError): e.child('market', learner_side)
                    failed = e.previous._document(files['case-' + name + '.failed.local.json'])
                    self.assertEqual(failed['completed_turns'], 5)
                    self.assertEqual(len(cleared), 5)
                    self.assertNotIn('case-' + name + '.local.json', files)
                else:
                    e.child('market', learner_side)
                    result = e._case_result(files['case-' + name + '.local.json'],
                                            head=head, opponent='market', side=learner_side)
                    self.assertEqual((result['turns'], result['callbacks'], result['teacher_side']),
                                     (719, 719, teacher_side))
                    self.assertEqual(len(cleared), 719)
            self.assertTrue(seen_sides and set(seen_sides) == {teacher_side})

        for side in (0, 1):
            exercise(False, side)
            exercise(True, side)


class CliTests(unittest.TestCase):
    def test_invalid_argument_and_exception_do_not_echo_private_text(self):
        secret = 'synthetic_private_location'
        for args in ([secret], ['child', secret, '0'], ['run', secret]):
            output = io.StringIO()
            with self.subTest(args=args), patch.object(sys, 'argv', ['experiment', *args]), redirect_stderr(output):
                with self.assertRaises(SystemExit) as caught: e.cli()
            self.assertEqual(caught.exception.code, 1)
            self.assertNotIn(secret, output.getvalue())
        output = io.StringIO()
        with patch.object(e, 'main', side_effect=FileNotFoundError(secret)), redirect_stderr(output):
            with self.assertRaises(SystemExit) as caught: e.cli()
        self.assertEqual(caught.exception.code, 1)
        self.assertNotIn(secret, output.getvalue())


if __name__ == '__main__':
    unittest.main()
