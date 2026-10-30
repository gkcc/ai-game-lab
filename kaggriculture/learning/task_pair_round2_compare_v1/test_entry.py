"""In-memory tests for round2 registration, attempt and failure boundaries."""
import copy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import entry


def typed_equal(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(typed_equal(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    return a == b


def registrations():
    old = dict(engine_commit='engine-commit', engine_sha256='engine-sha',
               previous_training_result_commit='training', historical_panel_result_commit='panel',
               public_sources={'old/source.py': 'old-digest', 'review/PROGRESS.md': 'old-progress'},
               limits=dict(initial_available_bytes=1, between_available_bytes=1,
                           campaign_wall_seconds=10800,
                           job=dict(min_available_bytes=1, max_rss_bytes=2)))
    new = copy.deepcopy(old)
    new['public_sources'].update({name: 'new-digest' for name in entry.EXTRA_REQUIRED})
    new['public_sources']['review/PROGRESS.md'] = 'updated-progress'
    new['limits']['initial_available_bytes'] = entry.INITIAL
    new['limits']['between_available_bytes'] = entry.BETWEEN
    new['limits']['job']['min_available_bytes'] = entry.JOB_MINIMUM
    return old, new


class RegistrationTests(unittest.TestCase):
    def test_help_exits_successfully_without_touching_an_attempt(self):
        with patch('sys.stdout', io.StringIO()), patch.object(entry, 'execute') as execute:
            self.assertEqual(entry.main(['--help']), 0)
        execute.assert_not_called()

    def test_termination_during_execution_never_looks_like_success(self):
        with patch.object(entry, 'execute', side_effect=SystemExit(0)):
            self.assertEqual(entry.main(['--game-root', 'unused',
                                         '--registration-commit', 'unused']), 1)

    def test_fresh_verifier_has_new_scope_and_retains_old_ancestry(self):
        auth = SimpleNamespace(REQUIRED=frozenset({'old'}), PRIOR=(('old-commit', ('old',)),))
        result = entry._adapt_auth(auth)
        self.assertIs(result, auth)
        self.assertEqual((auth.REGISTRATION, auth.STAGE_NAME, auth.SCHEMA),
                         (entry.REGISTRATION, entry.STAGE, entry.SCHEMA))
        self.assertEqual(auth.PANEL, entry.PANEL)
        self.assertEqual(auth.PANEL['opponents'], ['round2'])
        self.assertEqual(auth.EVALUATION, entry.EVALUATION)
        self.assertTrue(entry.EXTRA_REQUIRED.issubset(auth.REQUIRED))
        self.assertIn((entry.OLD_COMMIT, (entry.OLD_REGISTRATION,)), auth.PRIOR)
        self.assertIn((entry.OLD_FAILURE_COMMIT,
                       (entry.OLD_FAILURE_NOTE, entry.OLD_FAILURE_CERTIFICATE)), auth.PRIOR)
        self.assertEqual(len(entry.GAMES), 8)
        self.assertTrue(all('-round2-' in game for game in entry.GAMES))

    def test_original_source_and_exact_resource_inheritance(self):
        old, new = registrations()
        entry._validate_inheritance(new, old, typed_equal)
        for change in ('source', 'required', 'identity', 'initial', 'between', 'job_min', 'other_limit'):
            with self.subTest(change=change):
                candidate = copy.deepcopy(new)
                if change == 'source': candidate['public_sources']['old/source.py'] = 'different'
                if change == 'required': candidate['public_sources'].pop(next(iter(entry.EXTRA_REQUIRED)))
                if change == 'identity': candidate['engine_sha256'] = 'different'
                if change == 'initial': candidate['limits']['initial_available_bytes'] -= 1
                if change == 'between': candidate['limits']['between_available_bytes'] -= 1
                if change == 'job_min': candidate['limits']['job']['min_available_bytes'] -= 1
                if change == 'other_limit': candidate['limits']['job']['max_rss_bytes'] += 1
                with self.assertRaises(ValueError):
                    entry._validate_inheritance(candidate, old, typed_equal)

    def test_missing_snapshot_never_constructs_old_entry(self):
        with patch.object(entry, '_AUTH_HISTORY_ENTRY_BYTES', None, create=True):
            with self.assertRaises(ValueError):
                entry._historical()

    def test_save_callback_must_exist_before_an_attempt_is_available(self):
        old_registration, registration = registrations()
        identity = SimpleNamespace(_git=lambda *args: b'synthetic', _json=lambda raw: old_registration)
        ctx = dict(identity=identity, reg=registration, root=Path('synthetic-root'),
                   once_auth=SimpleNamespace(_typed_equal=typed_equal), limits=registration['limits'])
        old = SimpleNamespace(_context=Mock(return_value=ctx), _claim=Mock())
        coordinator = SimpleNamespace(Coordinator=lambda: None, limits_snapshot=lambda value: value)
        projection = SimpleNamespace(save_result=None)
        with patch.object(entry, '_historical', return_value=old), \
             patch.object(entry, '_check_package'), \
             patch.object(entry, '_load', side_effect=lambda name: coordinator if name == 'coordinator' else projection):
            with self.assertRaises(ValueError):
                entry._context('synthetic', 'a' * 40)
        old._claim.assert_not_called()


class FailureTests(unittest.TestCase):
    def test_failure_event_accepts_only_whitelisted_values_and_counters(self):
        coordinator = SimpleNamespace(FAILURE_PHASES={'launch'}, FAILURE_CATEGORIES={'MemoryError'})
        ctx = dict(coordinator=coordinator)
        good = dict(phase='launch', category='MemoryError', attempted=2, accepted=1)
        entry._event(ctx, good)
        self.assertEqual(ctx['coordinator_failure'], good)
        for bad in (dict(good, phase='private-path'), dict(good, category='secret'),
                    dict(good, accepted=3), dict(good, attempted=True), dict(good, extra='secret')):
            with self.subTest(bad=bad):
                ctx.pop('coordinator_failure', None)
                entry._event(ctx, bad)
                self.assertNotIn('coordinator_failure', ctx)

    def test_failure_records_do_not_copy_exception_text(self):
        ctx = dict(last_available_bytes=123)
        result = entry._failure_record(ctx, 'binding', ValueError('PRIVATE-DETAIL'))
        self.assertEqual(result['category'], 'invalid_input')
        self.assertEqual(result['last_available_bytes'], 123)
        self.assertNotIn('PRIVATE-DETAIL', str(result))
        self.assertFalse(result['retry_allowed'])

    def test_low_memory_fails_before_claim_or_game(self):
        once = SimpleNamespace(_starting_clock=lambda start, clock: lambda: start,
                               _check_entry_time=lambda *_: None)
        ctx = dict(once_entry=once, limits={'campaign_wall_seconds': 10})
        old = SimpleNamespace(_claim=Mock(), _start_child=Mock())
        err = io.StringIO()

        def low_memory(value):
            value['last_available_bytes'] = entry.INITIAL - 1
            return entry.INITIAL - 1

        with patch.object(entry, '_context', return_value=(old, ctx)), \
             patch.object(entry, '_sample', side_effect=low_memory), patch('sys.stderr', err):
            with self.assertRaises(MemoryError):
                entry.execute(game_root='unused', registration_commit='unused')
        old._claim.assert_not_called()
        old._start_child.assert_not_called()
        self.assertIn('"phase": "preflight"', err.getvalue())
        self.assertIn('"category": "memory_gate"', err.getvalue())

    def test_new_child_command_contains_only_round2_games(self):
        command = entry._command(Path('synthetic-root'), 'a' * 40, entry.GAMES[0])
        self.assertEqual(command[-2:], ['--game', entry.GAMES[0]])
        preflight = entry.command(Path('synthetic-root'), 'a' * 40, preflight=True)
        self.assertEqual(preflight[-1], '--preflight')
        self.assertNotIn('--game', preflight)
        with self.assertRaises(ValueError):
            entry._command(Path('synthetic-root'), 'a' * 40, 'dev-S02-market-0-baseline')
        with self.assertRaises(ValueError):
            entry.command(Path('synthetic-root'), 'a' * 40, game=entry.GAMES[0], preflight=True)


class PreflightTests(unittest.TestCase):
    def context(self):
        worker_source = b'synthetic-worker'
        plans = [dict(game=game, opponent='round2') for game in entry.GAMES]
        panel = SimpleNamespace(plans=lambda: plans, validate_plan=lambda plan: plan,
                                worker_sha=hashlib.sha256(worker_source).hexdigest())
        manifest = SimpleNamespace(panel=panel, opponent=lambda role: ('frozen-round2', 'digest'))
        previous = SimpleNamespace(opponent=lambda role: ('frozen-round2', 'digest'))
        old_dev = dict(state='failed_no_retry', prebinding_record_sha256='synthetic')
        current = dict(current_pair_dev_diagnostic=old_dev)
        bootstrap = SimpleNamespace(PREBIND='synthetic-binding', OLD_PANEL='artifacts/synthetic-panel',
            _private_status=lambda: (Path('synthetic-status'), current, old_dev, {}),
            _read=lambda *args: b'synthetic-stable', _verify_assets=Mock())
        once = SimpleNamespace(_verify_private_snapshot=Mock(), _fingerprint=lambda *args: b'fingerprint')
        bindings = SimpleNamespace(build=Mock(return_value=(manifest, previous, {})),
                                   verify_mapping=Mock(return_value=True))
        ctx = dict(bootstrap=bootstrap, identity=SimpleNamespace(_read=lambda path: worker_source),
                   once_entry=once, root=Path('synthetic-root'), reg={}, old_reg={}, freeze={},
                   once_auth=SimpleNamespace(_typed_equal=typed_equal), bindings=bindings,
                   modules={'history': object()}, public_check=lambda: True, time_check=lambda: True)
        return ctx, manifest

    def test_binding_checks_frozen_assets_without_claim_or_game(self):
        ctx, manifest = self.context()
        with patch.object(entry, '_unclaimed', return_value=True), \
             patch.object(entry, '_sample', return_value=entry.INITIAL):
            entry._preflight_binding(ctx)
        ctx['bindings'].build.assert_called_once()
        ctx['bootstrap']._verify_assets.assert_called_once_with(manifest, ctx['root'], ctx['modules']['history'])
        ctx['once_entry']._verify_private_snapshot.assert_called_once()

    def test_round2_opponent_mismatch_stops_before_assets(self):
        ctx, manifest = self.context()
        manifest.opponent = lambda role: ('different', 'digest')
        with patch.object(entry, '_unclaimed', return_value=True), \
             patch.object(entry, '_sample', return_value=entry.INITIAL):
            with self.assertRaises(ValueError):
                entry._preflight_binding(ctx)
        ctx['bootstrap']._verify_assets.assert_not_called()

    def test_public_preflight_does_not_claim_and_returns_safe_status(self):
        once = SimpleNamespace(_starting_clock=lambda start, clock: lambda: start,
                               _check_entry_time=lambda *_: None)
        old = SimpleNamespace(_claim=Mock())
        ctx = dict(once_entry=once, public_check=lambda: True, stage=Path('synthetic-root/stage'),
                   limits={'campaign_wall_seconds': 10, 'disk_free_floor_bytes': 4,
                           'per_game_reserve_bytes': 2})

        def sample(value):
            value['last_available_bytes'] = entry.INITIAL
            return entry.INITIAL

        with patch.object(entry, '_context', return_value=(old, ctx)), \
             patch.object(entry, '_sample', side_effect=sample), \
             patch.object(entry, '_unclaimed', return_value=True), \
             patch.object(entry, '_preflight_binding') as binding, \
             patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=10)):
            result = entry.public_preflight(game_root='unused', registration_commit='unused')
        binding.assert_called_once_with(ctx)
        old._claim.assert_not_called()
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['games_started'], 0)

    def test_any_diagnostic_marker_blocks_reusing_new_stage(self):
        stage = Path('synthetic-root/artifacts') / entry.STAGE
        ctx = dict(stage=stage, attempt_path=stage.with_name(stage.name + '.attempt.local.json'),
                   failure_path=stage.with_name(stage.name + '.failure.local.json'))
        with patch.object(entry.os.path, 'lexists', return_value=False):
            self.assertTrue(entry._unclaimed(ctx))
        marker = stage.with_name(stage.name + '.diagnostic.local.json')
        with patch.object(entry.os.path, 'lexists', side_effect=lambda path: Path(path) == marker):
            self.assertFalse(entry._unclaimed(ctx))


if __name__ == '__main__':
    unittest.main()
