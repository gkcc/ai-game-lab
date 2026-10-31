"""Focused synthetic tests; no seed generation or real game execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest import mock

import psutil

from . import projection
from .test_projection import fixture as projection_fixture


spec = importlib.util.spec_from_file_location('fresh_multi_entry_test', Path(__file__).with_name('entry.py'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class InMemoryParentFlow:
    """Exercise the parent coordinator without a filesystem, game or model."""

    def __init__(self, *, fail_second_cohort=False):
        self.cohorts, self.roles, _ = projection_fixture()
        self.manifests = {name: SimpleNamespace(cohort=name) for name in self.cohorts}
        self.binding, self.market_binding = b'synthetic-binding', b'synthetic-market-binding'
        self.files = {str(entry.PRIVATE / 'binding.local.json'): self.binding}
        self.events = []
        self.fail_second_cohort = fail_second_cohort

    def plans(self, manifest):
        return [dict(game=row['game'], seed_alias=row['seed_alias'], seed=row['seed'],
                     opponent=row['opponent'], side=row['side'], policy=row['arm'],
                     split='dev', parity_kind=None)
                for row in self.cohorts[manifest.cohort]['rows']]

    def write(self, path, data):
        key = str(path)
        if key in self.files:
            raise FileExistsError('Synthetic exclusive file already exists')
        self.files[key] = data

    def context(self):
        owner = self

        class Base:
            def read(self, path, *_):
                return owner.files[str(path)]

            _write = staticmethod(owner.write)
            decode = staticmethod(json.loads)
            validate_assets = staticmethod(lambda *_: True)
            typed_equal = staticmethod(lambda left, right: left == right)

        class Driver:
            def __init__(self, *, manifest, game_root, stage, commit, command_for, check):
                self.stage = stage

            def claim(self):
                return True

            def preflight_comparison(self, *_):
                return True

        class DriverModule:
            MAX_SOURCE = 123
            LocalDriver = Driver
            _no_links = staticmethod(lambda *_: None)
            _hash = staticmethod(lambda *_: 'synthetic-fixed-hash')
            _comparison_file = staticmethod(lambda stage, game, arm:
                stage / 'job-results' / (game + '-to-' + arm + '.comparison.local.json'))
            _read = staticmethod(lambda root, path, *_: owner.files.get(str(path), b'{}'))
            _write_new = staticmethod(lambda root, path, value, *_:
                owner.write(path, (json.dumps(value) + '\n').encode()))
            _replay = staticmethod(lambda *_: {})
            _owned_size = staticmethod(lambda *_: 0)

        class Coordinator:
            def run(self, *, manifest, limits, driver, check, authorize_game, claim, remaining):
                if claim() is not True:
                    raise AssertionError('Synthetic claim failed')
                owner.events.append(('cohort', manifest.cohort))
                for index, plan in enumerate(owner.plans(manifest)):
                    if authorize_game(plan) is not True:
                        raise AssertionError('Synthetic permit failed')
                    owner.events.append(('permit', manifest.cohort))
                    if owner.fail_second_cohort and manifest.cohort == 'C02' and index == 1:
                        raise RuntimeError('Synthetic second cohort failure')
                return copy.deepcopy(owner.cohorts[manifest.cohort])

        class Fresh:
            verify_inputs = staticmethod(lambda *_: True)
            verify_binding = staticmethod(lambda *_: (owner.binding, {}, {}))
            load_verified_market_binding = staticmethod(lambda: (owner.market_binding, {}))

        def compare(*_, **__):
            owner.events.append(('prefix', None))
            return dict(scope='same_complete_history', status='observed',
                        reason='actor_request_difference', step=11)

        return (Base(), {}, dict(driver=DriverModule(),
                                 acceptance=SimpleNamespace(_json=lambda raw, _: json.loads(raw))),
                SimpleNamespace(compare=compare), lambda: True, Fresh(),
                SimpleNamespace(Coordinator=Coordinator), projection,
                self.binding, dict(seed_roles={'F01': 101, 'F02': 202, 'F03': 303, 'F04': 404}),
                {}, self.market_binding, self.manifests,
                dict(job=dict(min_available_bytes=1), campaign_wall_seconds=10800,
                     initial_available_bytes=1, disk_free_floor_bytes=0))

    def run_with_mocks(self):
        original_stat, original_exists, original_symlink = Path.stat, Path.exists, Path.is_symlink
        stages = {str(entry.GAME / 'artifacts' / name) for name in entry.STAGES.values()}
        attempt = str(entry.PRIVATE / 'attempt.local.json')

        def fake_stat(path, *args, **kwargs):
            if str(path) in self.files:
                return SimpleNamespace(st_size=len(self.files[str(path)]))
            return original_stat(path, *args, **kwargs)

        def fake_exists(path):
            if str(path) in stages:
                return False
            if str(path) == attempt:
                return str(path) in self.files
            return original_exists(path)

        def fake_symlink(path):
            return False if str(path) in stages else original_symlink(path)

        def fake_finish(_commit, _binding, outcome):
            path = str(entry.PRIVATE / 'closed.local.json')
            if attempt not in self.files or path in self.files:
                return False
            self.files[path] = json.dumps({'outcome': outcome}).encode()
            return True

        with mock.patch.object(entry, '_context', return_value=self.context()), \
             mock.patch.object(entry, 'plans_for', side_effect=self.plans), \
             mock.patch.object(entry, '_game_evidence', side_effect=lambda **kw:
                 dict(game=kw['plan']['game'], cohort=kw['cohort'])), \
             mock.patch.object(entry, 'verify_comparison_receipt', return_value=True), \
             mock.patch.object(entry, '_finish_attempt', side_effect=fake_finish), \
             mock.patch.object(psutil, 'virtual_memory', return_value=SimpleNamespace(available=10 * 1024**3)), \
             mock.patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024**3)), \
             mock.patch.object(Path, 'stat', fake_stat), mock.patch.object(Path, 'exists', fake_exists), \
             mock.patch.object(Path, 'is_symlink', fake_symlink):
            return entry.run('a' * 40, 'b' * 64)


class EntryTests(unittest.TestCase):
    def test_total_deadline_does_not_reset_or_accept_a_clock_reversal_between_cohorts(self):
        times = iter((100.0, 150.0, 149.0))
        remaining = entry.shared_remaining(100.0, 1000.0, lambda: next(times))
        self.assertEqual(remaining(), 1000.0)
        self.assertEqual(remaining(), 950.0)
        with self.assertRaises(ValueError):
            remaining()

    def test_in_memory_parent_completes_two_cohorts_and_projects_once(self):
        flow = InMemoryParentFlow()
        result = flow.run_with_mocks()
        self.assertEqual((result['games'], result['official_replay_passes'],
                          result['comparisons'], result['prefix_comparisons']), (32, 32, 16, 16))
        self.assertEqual([event[1] for event in flow.events if event[0] == 'cohort'], ['C01', 'C02'])
        self.assertEqual(sum(event[0] == 'permit' for event in flow.events), 32)
        self.assertEqual(sum(event[0] == 'prefix' for event in flow.events), 16)
        permits = [json.loads(raw) for path, raw in flow.files.items() if path.endswith('.permit.local.json')]
        self.assertEqual({(item['cohort'], item['game']) for item in permits},
                         {(cohort, game) for cohort in entry.COHORTS for game in entry.GAMES})
        projected = json.loads(flow.files[str(entry.PRIVATE / 'projected-summary.local.json')])
        private = json.loads(flow.files[str(entry.PRIVATE / 'campaign-result.local.json')])
        self.assertEqual((len(projected['pairs']), len(private['receipts']),
                          len(private['comparison_receipts'])), (16, 32, 16))
        self.assertEqual(json.loads(flow.files[str(entry.PRIVATE / 'closed.local.json')])['outcome'], 'completed')

    def test_in_memory_second_cohort_failure_cannot_project_or_retry(self):
        flow = InMemoryParentFlow(fail_second_cohort=True)
        with self.assertRaises(RuntimeError):
            flow.run_with_mocks()
        self.assertEqual(sum(event == ('permit', 'C01') for event in flow.events), 16)
        self.assertEqual(sum(event == ('permit', 'C02') for event in flow.events), 2)
        self.assertNotIn(str(entry.PRIVATE / 'projected-summary.local.json'), flow.files)
        self.assertNotIn(str(entry.PRIVATE / 'campaign-result.local.json'), flow.files)
        self.assertEqual(json.loads(flow.files[str(entry.PRIVATE / 'closed.local.json')])['outcome'], 'failed')
        with self.assertRaises(ValueError):
            flow.run_with_mocks()
        self.assertEqual(sum(event[0] == 'permit' for event in flow.events), 18)

    def test_public_commitment_must_match_private_binding_digest(self):
        digest = 'a' * 64
        self.assertTrue(entry.validate_commitment({'binding_commitment_sha256': digest}, digest))
        for registered, supplied in (('b' * 64, digest), (digest, 'not-a-digest'), (None, digest)):
            with self.subTest(registered=registered), self.assertRaises(ValueError):
                entry.validate_commitment({'binding_commitment_sha256': registered}, supplied)

    def test_live_owner_requires_correct_ancestor_creation_time_and_deadline(self):
        commit, digest = 'a' * 40, 'b' * 64
        owner = json.loads(entry.owner_marker(commit=commit, binding_sha=digest, pid=10,
                                             created_us=123_000_000, deadline=200.0))
        class Proc:
            def __init__(self, pid, parent=None, exe='launcher', created=123.0):
                self.pid, self._parent, self._exe, self._created = pid, parent, exe, created
            def parent(self): return self._parent
            def is_running(self): return True
            def exe(self): return self._exe
            def create_time(self): return self._created
        coordinator = Proc(10)
        direct = Proc(20, coordinator)
        launcher = Proc(30, coordinator)
        indirect = Proc(40, launcher)
        def check(child, metadata=owner, now=150.0, samefile=lambda a, b: a == b):
            return entry.verify_live_owner(metadata, commit=commit, binding_sha=digest,
                current_pid=child.pid, process=lambda _pid: child, executable='launcher',
                samefile=samefile, now=now)
        self.assertTrue(check(direct))
        self.assertTrue(check(indirect))
        for bad, now in ((dict(owner, created_us=999), 150.0), (dict(owner, pid=11), 150.0),
                         (owner, 200.0)):
            with self.assertRaises(ValueError):
                check(indirect, bad, now)
        with self.assertRaises(ValueError):
            check(indirect, samefile=lambda *_: False)

    def test_failure_or_closed_marker_rejects_the_residual_authorization(self):
        stages = {key: entry.GAME / 'artifacts' / name for key, name in entry.STAGES.items()}
        self.assertTrue(entry.verify_open_attempt(stages, exists=lambda _path: False))
        failure = stages['C01'] / 'campaign-failure.local.json'
        closed = entry.PRIVATE / 'closed.local.json'
        for blocked in (failure, closed):
            with self.subTest(marker=blocked.name), self.assertRaises(ValueError):
                entry.verify_open_attempt(stages, exists=lambda path: path == blocked)

    def test_child_with_permit_cannot_start_after_any_cohort_failure_or_close(self):
        """Exercise the actual child branch with durable state held only in memory."""
        commit, digest, cohort, game = 'a' * 40, 'b' * 64, 'C01', entry.GAMES[0]
        stage = entry.GAME / 'artifacts' / entry.STAGES[cohort]
        stages = {key: entry.GAME / 'artifacts' / name for key, name in entry.STAGES.items()}
        binding, source = b'synthetic-binding', b'synthetic-worker-source'
        owner = entry.owner_marker(commit=commit, binding_sha=digest, pid=42,
                                   created_us=1234, deadline=time.monotonic() + 100)
        values = {
            entry.PRIVATE / 'owner.local.json': owner,
            entry.PRIVATE / 'attempt.local.json': entry.marker('attempt', commit=commit, binding_sha=digest),
            entry.PRIVATE / 'binding.local.json': binding,
            stage / 'manifest.local.json': binding,
            stage / 'job-results' / (game + '.permit.local.json'): entry.marker(
                'permit', commit=commit, binding_sha=digest, cohort=cohort, game=game),
        }
        real_gate = entry.verify_open_attempt

        def exercise(blocked):
            writes, worker_calls = [], []
            def read(path, *_args):
                return values[Path(path)]
            base = SimpleNamespace(read=read, decode=json.loads,
                _write=lambda path, data: writes.append((Path(path), data)), validate_assets=lambda *_: True)
            driver = SimpleNamespace(MAX_SOURCE=100, _hash=lambda *_: 'x' * 64,
                                     _no_links=lambda *_: None, _read=lambda *_: source)
            child = SimpleNamespace(run_one=lambda **kwargs: worker_calls.append(kwargs))
            manifest = SimpleNamespace(panel=SimpleNamespace(worker_sha=hashlib.sha256(source).hexdigest()))
            context = (base, {}, dict(driver=driver, acceptance=None, child=child), None, lambda: True,
                       None, None, None, binding, {}, {}, b'market',
                       {'C01': manifest, 'C02': manifest},
                       dict(campaign_wall_seconds=3600, job=dict(min_available_bytes=1)))

            def gate(actual_stages):
                return real_gate(actual_stages, exists=lambda path: path == blocked)

            with mock.patch.object(entry, '_context', return_value=context), \
                 mock.patch.object(entry, 'plans_for', return_value=[{'game': game}]), \
                 mock.patch.object(entry, 'verify_live_owner', return_value=True), \
                 mock.patch.object(entry, 'verify_open_attempt', side_effect=gate), \
                 mock.patch.object(psutil, 'virtual_memory', return_value=SimpleNamespace(available=2)), \
                 mock.patch.object(Path, 'is_dir', return_value=True):
                if blocked is None:
                    entry.run(commit, digest, cohort, game)
                else:
                    with self.assertRaises(ValueError):
                        entry.run(commit, digest, cohort, game)
            return writes, worker_calls

        writes, calls = exercise(None)
        self.assertEqual(len(writes), 2)
        self.assertEqual(len(calls), 1)
        for blocked in (stages['C01'] / 'campaign-failure.local.json',
                        stages['C02'] / 'campaign-failure.local.json',
                        entry.PRIVATE / 'closed.local.json'):
            with self.subTest(marker=blocked.name):
                writes, calls = exercise(blocked)
                self.assertEqual(writes, [])
                self.assertEqual(calls, [])

    def test_registered_limits_and_cohort_capacity_are_not_interchangeable(self):
        self.assertEqual(entry.validate_limits({'fresh_limits': dict(entry.FRESH_LIMITS)}), entry.FRESH_LIMITS)
        for key, value in (('max_games', 16), ('max_comparisons', 8), ('max_prefix_comparisons', 8), ('max_attempts', True),
                           ('cohort_output_bytes', 2 * 1024**3)):
            changed = dict(entry.FRESH_LIMITS, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.validate_limits({'fresh_limits': changed})
        legacy = dict(job={'min_available_bytes': 3}, max_games=8, max_replays=8,
                      max_comparisons=4, max_attempts=1, total_output_bytes=2 * 1024**3)
        coordinator = SimpleNamespace(limits_snapshot=lambda value: value)
        result = entry.cohort_limits(legacy, coordinator, dict(entry.FRESH_LIMITS))
        self.assertEqual((result['max_games'], result['max_comparisons'], result['total_output_bytes']),
                         (16, 8, 3 * 1024**3))
        self.assertEqual(legacy['max_games'], 8)

    def test_two_opponents_both_seats_and_two_arms_exclude_auxiliary(self):
        plans = [dict(game=f'dev-{alias}-{opponent}-{side}-{arm}', seed_alias=alias,
                      opponent=opponent, side=side, policy=arm)
                 for alias in entry.ALIASES for opponent in entry.OPPONENTS
                 for side in (0, 1) for arm in ('baseline', 'control', 'auxiliary')]
        panel = SimpleNamespace(plans=lambda: tuple(plans), validate_plan=lambda plan: dict(plan))
        chosen = entry.plans_for(SimpleNamespace(panel=panel))
        self.assertEqual(len(chosen), 16)
        self.assertEqual({p['policy'] for p in chosen}, {'baseline', 'control'})
        self.assertEqual({p['opponent'] for p in chosen}, {'market', 'round2'})
        self.assertEqual({p['side'] for p in chosen}, {0, 1})
        del plans[-1]
        del plans[-1]
        with self.assertRaises(ValueError):
            entry.plans_for(SimpleNamespace(panel=panel))

    def test_per_game_marker_binds_both_cohort_and_game(self):
        commit, digest, game = 'a' * 40, 'b' * 64, entry.GAMES[0]
        a = entry.marker('permit', commit=commit, binding_sha=digest, cohort='C01', game=game)
        b = entry.marker('permit', commit=commit, binding_sha=digest, cohort='C02', game=game)
        self.assertNotEqual(a, b)
        self.assertEqual(json.loads(a)['cohort'], 'C01')
        with self.assertRaises(ValueError):
            entry.marker('permit', commit=commit, binding_sha=digest, cohort='C03', game=game)
        with self.assertRaises(ValueError):
            entry.marker('attempt', commit=commit, binding_sha=digest, cohort='C01', game=game)

    def test_comparison_receipt_must_bind_both_replays_traces_and_payload(self):
        left = dict(game='left', replay_sha256='a' * 64, trace_sha256='b' * 64)
        right = dict(game='right', replay_sha256='c' * 64, trace_sha256='d' * 64)
        comparison = {'seed_alias': 'S02', 'side': 0, 'result': {'stop_reason': 'choice_difference'}}
        receipt = dict(schema='task-pair-dev-comparison-v1', reference_game='left', candidate_game='right',
                       reference_replay_sha256=left['replay_sha256'], candidate_replay_sha256=right['replay_sha256'],
                       reference_trace_sha256=left['trace_sha256'], candidate_trace_sha256=right['trace_sha256'],
                       comparison=copy.deepcopy(comparison))
        base = SimpleNamespace(typed_equal=lambda a, b: a == b)
        self.assertTrue(entry.verify_comparison_receipt(base, receipt, comparison, left, right))
        for field in ('candidate_game', 'reference_replay_sha256', 'candidate_trace_sha256'):
            changed = dict(receipt, **{field: 'wrong'})
            with self.subTest(field=field), self.assertRaises(ValueError):
                entry.verify_comparison_receipt(base, changed, comparison, left, right)
        changed = copy.deepcopy(receipt)
        changed['comparison']['result']['stop_reason'] = 'frame_difference'
        with self.assertRaises(ValueError):
            entry.verify_comparison_receipt(base, changed, comparison, left, right)


if __name__ == '__main__':
    unittest.main()
