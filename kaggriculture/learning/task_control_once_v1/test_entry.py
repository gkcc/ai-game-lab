"""Synthetic no-game checks for registration, markers and sanitized errors."""
from contextlib import redirect_stderr
import io
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import entry
from . import auth


class EntryTests(unittest.TestCase):
    def test_unknown_arguments_do_not_echo_private_values(self):
        marker = 'SYNTHETIC_PRIVATE_ARGUMENT'
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as stopped:
            entry.main(['--unexpected', marker])
        self.assertNotIn(marker, str(stopped.exception) + stderr.getvalue())

    def test_system_exit_zero_is_not_a_normal_entry_return(self):
        with patch.object(entry, 'execute', side_effect=SystemExit(0)), self.assertRaises(SystemExit) as stopped:
            entry.main(['--game-root', 'synthetic', '--registration-commit', 'a' * 40])
        self.assertNotIn(stopped.exception.code, (0, None))

    def test_system_exit_message_does_not_echo_private_value(self):
        marker = 'SYNTHETIC_PRIVATE_EXIT_VALUE'
        with patch.object(entry, 'execute', side_effect=SystemExit(marker)), self.assertRaises(SystemExit) as stopped:
            entry.main(['--game-root', 'synthetic', '--registration-commit', 'a' * 40])
        self.assertNotIn(marker, str(stopped.exception))

    def test_child_command_allows_only_one_registered_game(self):
        with self.assertRaises(ValueError):
            entry._command(Path('synthetic'), 'a' * 40, 'other-game')
        command = entry._command(Path('synthetic'), 'a' * 40, entry.GAME)
        self.assertEqual(command[-2:], ['--game', entry.GAME])
        self.assertEqual(command[1:4], ['-I', '-B', '-c'])

    def test_game_permit_can_only_be_consumed_once(self):
        stage = Path('synthetic-stage')
        storage = {}
        ctx = dict(stage=stage, plan={'synthetic': 'plan'}, commit='a' * 40,
                   attempt_raw=b'attempt', manifest_raw=b'manifest', failure_path=Path('synthetic-failure'),
                   public_check=lambda: True)
        ctx['modules'] = {'driver': SimpleNamespace(_read=lambda _root, path, _limit: storage[str(path)])}
        def exists(path):
            return str(path) in storage
        def write(_ctx, relative, data):
            path = str(stage / relative)
            if path in storage:
                raise FileExistsError('synthetic')
            storage[path] = data
        with patch.object(entry, '_check', return_value=True), \
             patch.object(entry, '_attempt_check', return_value=True), patch.object(Path, 'is_dir', return_value=True), \
             patch.object(entry.os.path, 'lexists', side_effect=exists), \
             patch.object(entry, '_write_stage', side_effect=write):
            entry._authorize(ctx, ctx['plan'])
            entry._start_child(ctx)
            with self.assertRaises(ValueError):
                entry._start_child(ctx)
            entry._consume(ctx)
            with self.assertRaises(ValueError):
                entry._consume(ctx)
            with self.assertRaises(ValueError):
                entry._authorize(ctx, ctx['plan'])

    def test_bad_public_limits_stop_before_private_status(self):
        root = entry.PUBLIC_ROOT.parent.parent / 'kaggriculture'
        if not root.exists():
            self.skipTest('This checkout has no separate game root')
        bootstrap = SimpleNamespace(_absolute=lambda _value: root,
                                    _runtime=lambda _root, _reg: {},
                                    _private_status=lambda: self.fail('private status was opened'))
        identity = SimpleNamespace()
        auth = SimpleNamespace(verify_public=lambda *_args, **_kwargs: ({'limits': {}}, {}, {}))
        runner = SimpleNamespace(limits_snapshot=lambda _value: (_ for _ in ()).throw(ValueError('bad limit')))
        def load(name):
            return runner if name == 'runner' else SimpleNamespace()
        flags = SimpleNamespace(isolated=1, dont_write_bytecode=1)
        with patch.object(entry, '_snapshots', return_value=(identity, bootstrap, auth)), \
             patch.object(entry, '_check_package'), patch.object(entry, '_load', side_effect=load), \
             patch.object(entry.sys, 'flags', flags):
            with self.assertRaises(ValueError):
                entry._build_context(root, 'a' * 40)

    def test_changed_running_limits_fail_the_recheck(self):
        ctx = dict(precheck=lambda: True, auth=auth,
                   runner=SimpleNamespace(limits_snapshot=lambda value: value),
                   reg={'limits': {'max_games': 1}}, limits={'max_games': True},
                   stage=Path('synthetic-stage'), failure_path=Path('synthetic-failure'))
        with patch.object(entry, '_attempt_check', return_value=True), \
             patch.object(entry.os.path, 'lexists', return_value=False):
            self.assertFalse(entry._check(ctx, False))

    def test_pre_stage_failure_is_recorded_without_guessing_acceptance(self):
        original = RuntimeError('synthetic failure')
        def fail_driver(**_kwargs):
            raise original
        ctx = dict(modules={'driver': SimpleNamespace(LocalDriver=fail_driver)},
                   manifest=object(), root=Path('synthetic-root'), stage=Path('synthetic-stage'),
                   commit='a' * 40, attempt_raw=b'attempt', limits={'campaign_wall_seconds': 60})
        recorded = []
        with patch.object(entry, '_build_context', return_value=ctx), \
             patch.object(entry, '_claim_attempt', side_effect=lambda value: value.update(_attempt_created=True)), \
             patch.object(entry, '_bind_private'), \
             patch.object(entry, '_attempt_failure', side_effect=lambda value: recorded.append(entry._failure_record(value))):
            with self.assertRaises(RuntimeError) as stopped:
                entry.execute(game_root='unused', registration_commit='a' * 40)
        self.assertIs(stopped.exception, original)
        self.assertEqual(len(recorded), 1)
        self.assertIsNone(recorded[0]['accepted_games'])
        self.assertFalse(recorded[0]['retry_allowed'])

    def test_campaign_clock_includes_prior_entry_work(self):
        clock = entry._starting_clock(100.0, lambda: 107.0)
        self.assertEqual(clock(), 100.0)
        self.assertEqual(clock(), 107.0)
        readings = iter((80.0, 70.0))
        clock = entry._starting_clock(0.0, lambda: next(readings))
        self.assertEqual((clock(), clock()), (0.0, 80.0))
        with self.assertRaises(ValueError):
            clock()
        sampled = entry._starting_clock(100.0, lambda: 107.0)
        self.assertEqual((sampled(), sampled()), (100.0, 107.0))
        coordinator = entry._coordinator_clock(100.0, sampled)
        self.assertEqual((coordinator(), coordinator()), (100.0, 107.0))

    def test_return_marker_requires_coordinator_to_return(self):
        class Coordinator:
            def __init__(self, result):
                self.result = result
            def run(self, **_kwargs):
                if isinstance(self.result, BaseException):
                    raise self.result
                return self.result
        success = dict(accepted=1, official_replay_passes=1, promotion=False, observed_outer_exit=False)
        for outcome in (success, SystemExit(0)):
            with self.subTest(cancel=isinstance(outcome, BaseException)):
                driver = SimpleNamespace(claim=lambda: True, launch=lambda *_args: None)
                ctx = dict(modules={'driver': SimpleNamespace(LocalDriver=lambda **_kwargs: driver)},
                           manifest=object(), root=Path('synthetic-root'), stage=Path('synthetic-stage'),
                           commit='a' * 40, plan={'synthetic': True}, limits={'campaign_wall_seconds': 60}, attempt_raw=b'attempt',
                           runner=SimpleNamespace(Coordinator=lambda: Coordinator(outcome)))
                written = []
                with patch.object(entry, '_build_context', return_value=ctx), \
                     patch.object(entry, '_claim_attempt', side_effect=lambda value: value.update(_attempt_created=True)), \
                     patch.object(entry, '_bind_private'), \
                     patch.object(entry, '_check', return_value=True), \
                     patch.object(entry, '_attempt_failure'), \
                     patch.object(entry, '_write_stage', side_effect=lambda _ctx, name, _data: written.append(name)):
                    if isinstance(outcome, BaseException):
                        with self.assertRaises(SystemExit):
                            entry.execute(game_root='unused', registration_commit='a' * 40)
                        self.assertNotIn('control-returned.local.json', written)
                    else:
                        entry.execute(game_root='unused', registration_commit='a' * 40)
                        self.assertIn('control-returned.local.json', written)

    def test_entry_clock_rejects_expired_or_reversed_time(self):
        entry._check_entry_time(10.0, 5, lambda: 14.9)
        for now in (9.0, 15.0, float('nan')):
            with self.subTest(now=now), self.assertRaises(ValueError):
                entry._check_entry_time(10.0, 5, lambda: now)

    def test_private_binding_failure_consumes_attempt_and_records_unknown(self):
        ctx = dict(stage=Path('synthetic-stage'), commit='a' * 40, attempt_raw=b'attempt',
                   limits={'campaign_wall_seconds': 60})
        calls = []
        def claim(value):
            calls.append('claim')
            value['_attempt_created'] = True
        def bind(_value):
            calls.append('private')
            raise ValueError('synthetic binding failure')
        with patch.object(entry, '_build_context', return_value=ctx), \
             patch.object(entry, '_claim_attempt', side_effect=claim), \
             patch.object(entry, '_bind_private', side_effect=bind), \
             patch.object(entry, '_attempt_failure', side_effect=lambda _value: calls.append('failure')):
            with self.assertRaises(ValueError):
                entry.execute(game_root='unused', registration_commit='a' * 40)
        self.assertEqual(calls, ['claim', 'private', 'failure'])

    def test_expired_entry_stops_before_private_binding_after_claim(self):
        ctx = dict(stage=Path('synthetic-stage'), commit='a' * 40, attempt_raw=b'attempt',
                   limits={'campaign_wall_seconds': 5})
        samples = iter((100.0, 106.0))
        with patch.object(entry, '_build_context', return_value=ctx), \
             patch.object(entry, '_claim_attempt', side_effect=lambda value: value.update(_attempt_created=True)), \
             patch.object(entry, '_bind_private') as private, patch.object(entry, '_attempt_failure') as failure:
            with self.assertRaises(ValueError):
                entry.execute(game_root='unused', registration_commit='a' * 40, clock=lambda: next(samples))
        private.assert_not_called()
        failure.assert_called_once_with(ctx)

    def test_expired_binding_recheck_stops_before_private_status_read(self):
        reads = []
        bootstrap = SimpleNamespace(_private_status=lambda: reads.append('private'))
        samples = iter((True, False))
        ctx = dict(time_check=lambda: next(samples), public_check=lambda: True,
                   identity=object(), bootstrap=bootstrap, bindings=object(),
                   reg={}, old_reg={}, freeze={}, root=Path('synthetic-root'))
        with patch.object(entry, '_not_failed', return_value=True), \
             patch.object(entry, '_attempt_check', return_value=True), self.assertRaises(ValueError):
            entry._bind_private(ctx)
        self.assertEqual(reads, [])

    def test_expired_child_stops_before_private_binding(self):
        ctx = dict(limits={'campaign_wall_seconds': 5})
        samples = iter((100.0, 106.0))
        with patch.object(entry, '_build_context', return_value=ctx), \
             patch.object(entry, '_start_child') as started, patch.object(entry, '_bind_private') as private:
            with self.assertRaises(ValueError):
                entry.execute(game_root='unused', registration_commit='a' * 40,
                              game=entry.GAME, clock=lambda: next(samples))
        started.assert_called_once_with(ctx)
        private.assert_not_called()

    def test_unwritable_attempt_is_marked_for_best_effort_failure_record(self):
        ctx = dict(public_check=lambda: True, stage=Path('synthetic-stage'),
                   attempt_path=Path('synthetic-attempt'), failure_path=Path('synthetic-failure'),
                   _attempt_created=False, _attempt_open_failed=False)
        with patch.object(entry.os.path, 'lexists', return_value=False), \
             patch.object(Path, 'open', side_effect=PermissionError('synthetic denial')):
            with self.assertRaises(PermissionError):
                entry._claim_attempt(ctx)
        self.assertFalse(ctx['_attempt_created'])
        self.assertTrue(ctx['_attempt_open_failed'])

    def test_private_status_snapshot_must_match_the_authenticated_objects(self):
        bootstrap = SimpleNamespace(_json=lambda raw: json.loads(raw))
        bound = b'{"identity":"a"}'
        digest = hashlib.sha256(bound).hexdigest()
        entry._verify_private_snapshot(bootstrap, auth, {'state': 1}, {'identity': 'a'},
                                       b'{"state":1}', bound, digest)
        with self.assertRaises(ValueError):
            entry._verify_private_snapshot(bootstrap, auth, {'state': 1}, {'identity': 'a'},
                                           b'{"state":true}', bound, digest)
        with self.assertRaises(ValueError):
            entry._verify_private_snapshot(bootstrap, auth, {'state': 1}, {'identity': 'a'},
                                           b'{"state":1}', b'{"identity": "a"}', digest)

    def test_failed_child_cannot_bind_or_reuse_a_remaining_permit(self):
        stage, failure = Path('synthetic-stage'), Path('synthetic-failure')
        permit = stage / entry._marker_name('permit')
        for failed in (failure, stage / 'campaign-failure.local.json'):
            ctx = dict(stage=stage, failure_path=failure, public_check=lambda: True,
                       commit='a' * 40, attempt_raw=b'attempt')
            present = {str(permit), str(failed)}
            with self.subTest(), patch.object(entry, '_build_context', return_value=ctx), \
                 patch.object(entry, '_attempt_check', return_value=True), \
                 patch.object(Path, 'is_dir', return_value=True), \
                 patch.object(entry.os.path, 'lexists', side_effect=lambda path: str(path) in present), \
                 patch.object(entry, '_bind_private') as private:
                with self.assertRaises(ValueError):
                    entry.execute(game_root='unused', registration_commit='a' * 40, game=entry.GAME)
                private.assert_not_called()

    def test_child_binding_failure_cannot_reuse_the_same_permit(self):
        stage, failure = Path('synthetic-stage'), Path('synthetic-failure')
        ctx = dict(stage=stage, failure_path=failure, public_check=lambda: True,
                   commit='a' * 40, attempt_raw=b'attempt', limits={'campaign_wall_seconds': 60})
        present = {str(stage / entry._marker_name('permit'))}
        def write(_ctx, relative, _data):
            name = str(stage / relative)
            if name in present:
                raise FileExistsError('synthetic')
            present.add(name)
        with patch.object(entry, '_build_context', return_value=ctx), \
             patch.object(entry, '_attempt_check', return_value=True), patch.object(Path, 'is_dir', return_value=True), \
             patch.object(entry.os.path, 'lexists', side_effect=lambda path: str(path) in present), \
             patch.object(entry, '_write_stage', side_effect=write), \
             patch.object(entry, '_bind_private', side_effect=ValueError('synthetic')) as private:
            for _ in range(2):
                with self.assertRaises(ValueError):
                    entry.execute(game_root='unused', registration_commit='a' * 40, game=entry.GAME)
            self.assertEqual(private.call_count, 1)


if __name__ == '__main__':
    unittest.main()
