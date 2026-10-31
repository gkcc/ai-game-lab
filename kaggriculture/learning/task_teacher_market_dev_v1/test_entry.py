"""Focused invariants for the teacher-only market diagnostic; no real games."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location('teacher_market_entry_test', Path(__file__).with_name('entry.py'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def fixture():
    panel = SimpleNamespace(seeds=(12, 34))
    plan = entry.plans(panel)[0]
    job = dict(plan, bot_sha256='a' * 64, rival_sha256='b' * 64, task_checkpoint_sha256=None)
    row = dict(job, passed=True, states=720, action_turns=719, status=['DONE', 'DONE'], agent_errors=0,
               crop_checkpoint_sha256='c' * 64, engine_sha256='d' * 64, initial_banks=[100, 100],
               banks=[110, 108], rewards=[110, 108], bank=110, opponent_bank=108,
               initial_bank=100, opponent_initial_bank=100, net_profit=10, opponent_net_profit=8,
               margin=2, score=1., trace_sha256='e' * 64, replay_gzip_sha256='f' * 64,
               semantic_replay_sha256='1' * 64, semantic_replay_sha256_v2='2' * 64)
    counts = dict(task_forwards=0, task_rows=0, crop_forwards=4, controller_calls=719)
    official = dict(passed=True, official_transitions_reexecuted=719,
                    both_public_and_private_states_match_each_step=True, inventory_order_checked_each_step=True,
                    engine_sha256='d' * 64, semantic_sha256='1' * 64, semantic_sha256_v2='2' * 64,
                    final_rewards=[110, 108])
    audit = dict(game=plan['game'], passed=True, errors=0, actor_actions_reproduced=719,
                 rival_actions_reproduced=719, trace_turns_reproduced=719, trace_sha256='e' * 64,
                 replay_sha256='f' * 64, official=official, primary_model_counts=counts,
                 live_model_counts=copy.deepcopy(counts))
    complete = dict(game=plan['game'], passed=True, row_sha256='3' * 64, certificate_sha256='4' * 64)
    storage = dict(game=plan['game'], passed=True, error_kind=None, native_output_accounted_by='NUL scope',
                   output=dict(limit_bytes=100, used_bytes=30, remaining_bytes=70, refused=0))
    bundle = dict(row=row, audit=audit, complete=complete, storage=storage)
    hashes = dict(row='3' * 64, audit='4' * 64, complete='5' * 64, storage='6' * 64)
    return plan, job, bundle, hashes


def accept(plan, job, bundle, hashes):
    return entry.verify_bundle(plan, job, bundle, True, crop_sha='c' * 64, engine_sha='d' * 64,
        storage_limit=100, native_scope='NUL scope', receipt_hashes=hashes, verify_files=lambda _p, _r: True)


class TeacherIdentityTests(unittest.TestCase):
    def test_worker_adapter_preserves_teacher_identity_and_restores_globals(self):
        plan, job, _bundle, _hashes = fixture()
        stage = Path.cwd().resolve() / 'synthetic-teacher-stage'
        config = {'crop': 'fixed'}
        original_verify = lambda _path: None
        original_identity = lambda _path, _plan, _config: (None, None)
        module = SimpleNamespace(verify=original_verify, task_identity=original_identity,
                                 read=lambda _path: None, stable=lambda value: value,
                                 write_new=lambda _path, _value: None)
        worker = SimpleNamespace(_load_snapshot=lambda *_args: module,
                                 _read_job=lambda _path: dict(job), NATIVE_OUTPUT_SCOPE='NUL scope')
        panel = SimpleNamespace(worker_sha='a' * 64, worker_config=lambda: dict(config))
        sys.path.insert(0, str(entry.GAME))
        try:
            with entry.bound_worker(worker, b'synthetic', SimpleNamespace(panel=panel),
                                    entry.GAME, stage, plan, lambda: True) as (bound, actual, expected):
                self.assertEqual(bound.verify(stage), config)
                self.assertEqual(bound.task_identity(stage, plan, actual), (None, None))
                expected[0] = job
                self.assertEqual(bound.read(stage / 'job-inputs' / (plan['game'] + '.json')), job)
                with self.assertRaises(ValueError):
                    bound.task_identity(stage, dict(plan, policy='baseline'), actual)
            self.assertIs(module.verify, original_verify)
            self.assertIs(module.task_identity, original_identity)
        finally:
            sys.path.remove(str(entry.GAME))

    def test_none_is_required_for_the_teacher_job(self):
        plan, job, _bundle, _hashes = fixture()
        self.assertEqual(entry.validate_plan(job, plan, job=True), job)
        job['task_checkpoint_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            entry.validate_plan(job, plan, job=True)

    def test_changed_seat_or_seed_is_rejected(self):
        plan, job, _bundle, _hashes = fixture()
        for key, changed in (('side', 1), ('seed', 35), ('policy', 'baseline')):
            altered = dict(job, **{key: changed})
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.validate_plan(altered, plan, job=True)

    def test_real_game_receipt_invariants(self):
        plan, job, bundle, hashes = fixture()
        self.assertEqual(accept(plan, job, bundle, hashes), bundle['row'])
        changes = (
            ('task_forwards', 1), ('task_rows', 1), ('crop_forwards', 0), ('controller_calls', 718),
        )
        for name, value in changes:
            modified = copy.deepcopy(bundle)
            modified['audit']['primary_model_counts'][name] = value
            modified['audit']['live_model_counts'][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                accept(plan, job, modified, hashes)
        modified = copy.deepcopy(bundle)
        modified['audit']['official']['official_transitions_reexecuted'] = 718
        with self.assertRaises(ValueError):
            accept(plan, job, modified, hashes)
        modified = copy.deepcopy(bundle)
        modified['complete']['row_sha256'] = '7' * 64
        with self.assertRaises(ValueError):
            accept(plan, job, modified, hashes)
        modified = copy.deepcopy(bundle)
        modified['row']['net_profit'] = 11
        with self.assertRaises(ValueError):
            accept(plan, job, modified, hashes)
        modified = copy.deepcopy(bundle)
        del modified['row']['semantic_replay_sha256']
        del modified['audit']['official']['semantic_sha256']
        with self.assertRaises(ValueError):
            accept(plan, job, modified, hashes)

    def test_entire_trace_must_remain_teacher_and_crop_model(self):
        plan, _job, bundle, _hashes = fixture()
        turns = [dict(task_checkpoint_sha256=None, checkpoint_sha256='c' * 64,
                      records=[dict(source='teacher', chosen='a', teacher_label='a')],
                      crop_records=[dict(source='model')]) for _ in range(719)]
        class Driver:
            @staticmethod
            def _trace(_stage, _path, _expected):
                yield from turns
        result = entry.trace_summary(Driver, Path('stage'), plan, bundle['row'], 'c' * 64)
        self.assertEqual(result['task_teacher_choices'], 719)
        turns[-1]['records'][0]['chosen'] = 'b'
        with self.assertRaises(ValueError):
            entry.trace_summary(Driver, Path('stage'), plan, bundle['row'], 'c' * 64)
        turns[-1]['records'][0]['chosen'] = 'a'
        del turns[0]['task_checkpoint_sha256']
        with self.assertRaises(ValueError):
            entry.trace_summary(Driver, Path('stage'), plan, bundle['row'], 'c' * 64)

    def test_explicit_prefix_budget_is_distinct_from_legacy_comparison_capacity(self):
        limits = dict(entry.TEACHER_LIMITS)
        self.assertEqual(entry.validate_teacher_limits({'teacher_limits': limits}), limits)
        for key, bad in (('max_teacher_games', 8), ('max_official_replays', 8),
                         ('max_prefix_comparisons', 4), ('max_attempts', True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.validate_teacher_limits({'teacher_limits': dict(limits, **{key: bad})})


class ProjectionTests(unittest.TestCase):
    def test_teacher_minus_student_and_unknown_prefix_are_preserved(self):
        plans = entry.plans(SimpleNamespace(seeds=(12, 34)))
        rows = [dict(seed_alias=p['seed_alias'], side=p['side'], net_profit=10,
                     opponent_net_profit=8, margin=2, score=1.) for p in plans]
        economic = dict(own_net_profit=7, opponent_net_profit=9, margin=-2, score=0.)
        previous = {'pairs': [dict(seed_alias=entry.ALIASES[p['seed_alias']], side=p['side'],
                                   reference=economic, candidate=economic) for p in plans]}
        unknown = dict(scope='same_complete_history', status='unknown', reason='frame_difference', step=None)
        prefixes = {(p['seed_alias'], p['side'], arm): unknown for p in plans for arm in ('baseline', 'control')}
        projected = entry.project(rows, previous, prefixes, 'a' * 40)
        self.assertEqual(len(projected['pairs']), 8)
        self.assertEqual(projected['mean_teacher_minus_student']['baseline']['own_net_profit'], 3)
        self.assertEqual(projected['mean_teacher_minus_student']['control']['margin'], 4)
        self.assertTrue(all(p['complete_request_prefix']['status'] == 'unknown' for p in projected['pairs']))
        self.assertFalse(projected['independent_improvement'])


if __name__ == '__main__':
    unittest.main()
