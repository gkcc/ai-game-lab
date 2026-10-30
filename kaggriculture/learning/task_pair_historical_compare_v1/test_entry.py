"""Synthetic regressions for the registered eight-game entry contract."""
from contextlib import redirect_stderr
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import auth, entry
from learning.task_control_once_v1 import auth as once_auth


class EntryTests(unittest.TestCase):
    def test_registered_game_coordinates_and_no_unregistered_command(self):
        self.assertEqual(len(entry.GAMES), 8)
        self.assertEqual(len(set(entry.GAMES)), 8)
        for game in entry.GAMES:
            command = entry._command(Path('synthetic-root'), 'a' * 40, game)
            self.assertEqual(command[-2:], ['--game', game])
            self.assertEqual(command[1:4], ['-I', '-B', '-c'])
        for game in ('dev-S02-round2-0-control', 'dev-S02-market-0-auxiliary',
                     'dev-S03-market-2-baseline', 'unregistered'):
            with self.subTest(game=game), self.assertRaises(ValueError):
                entry._command(Path('synthetic-root'), 'a' * 40, game)

    def test_each_game_permit_is_single_use_and_bound_to_game(self):
        stage = Path('synthetic-stage')
        storage = {}
        games = entry.GAMES[:2]
        plans = [{'game': game} for game in games]
        ctx = dict(stage=stage, plans=plans, commit='a' * 40, attempt_raw=b'attempt',
                   manifest_raw=b'manifest', failure_path=Path('synthetic-failure'),
                   public_check=lambda: True, manifest=object(), source=b'synthetic', root=Path('synthetic-root'))
        calls = []
        ctx['modules'] = {
            'driver': SimpleNamespace(_read=lambda _root, path, _limit: storage[str(path)]),
            'child': SimpleNamespace(run_one=lambda **kwargs: calls.append(kwargs['plan']['game'])),
        }
        def exists(path):
            return str(path) in storage
        def write(_ctx, relative, value):
            name = str(stage / relative)
            if name in storage:
                raise FileExistsError('synthetic')
            storage[name] = entry._json(value)
        with patch.object(entry, '_check', return_value=True), \
             patch.object(entry, '_attempt_check', return_value=True), patch.object(Path, 'is_dir', return_value=True), \
             patch.object(entry.os.path, 'lexists', side_effect=exists), \
             patch.object(entry, '_write_stage', side_effect=write):
            entry._authorize(ctx, plans[0])
            with self.assertRaises(ValueError):
                entry._start_child(ctx, games[1])
            entry._start_child(ctx, games[0])
            with self.assertRaises(ValueError):
                entry._start_child(ctx, games[0])
            entry._run_child(ctx, games[0])
            self.assertEqual(calls, [games[0]])
            with self.assertRaises(ValueError):
                entry._run_child(ctx, games[0])
            with self.assertRaises(ValueError):
                entry._authorize(ctx, plans[0])
            entry._authorize(ctx, plans[1])
            entry._start_child(ctx, games[1])
            entry._run_child(ctx, games[1])
            self.assertEqual(calls, list(games))

    def test_unknown_argument_and_system_exit_do_not_echo_private_values(self):
        marker = 'SYNTHETIC_PRIVATE_VALUE'
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as stopped:
            entry.main(['--unexpected', marker])
        self.assertNotIn(marker, str(stopped.exception) + stderr.getvalue())
        with patch.object(entry, 'execute', side_effect=SystemExit(marker)), self.assertRaises(SystemExit) as stopped:
            entry.main(['--game-root', 'synthetic', '--registration-commit', 'a' * 40])
        self.assertNotIn(marker, str(stopped.exception))
        self.assertNotIn(stopped.exception.code, (0, None))

    def test_private_status_read_does_not_continue_after_deadline(self):
        calls = []
        bootstrap = SimpleNamespace(_private_status=lambda: calls.append('private'))
        checks = iter((True, False))
        ctx = dict(time_check=lambda: next(checks), public_check=lambda: True,
                   identity=object(), bootstrap=bootstrap, once_entry=object(),
                   reg={}, old_reg={}, freeze={}, root=Path('synthetic-root'))
        with patch.object(entry, '_not_failed', return_value=True), \
             patch.object(entry, '_attempt_check', return_value=True), self.assertRaises(ValueError):
            entry._bind(ctx)
        self.assertEqual(calls, [])

    def test_public_preflight_never_binds_private_inputs_or_starts_a_game(self):
        ctx = dict(public_check=lambda: True, stage=Path('synthetic-stage'),
                   attempt_path=Path('synthetic-attempt'), failure_path=Path('synthetic-failure'))
        with patch.object(entry, '_context', return_value=ctx), \
             patch.object(entry.os.path, 'lexists', return_value=False), \
             patch.object(entry, '_bind') as bind, patch.object(entry, '_claim') as claim:
            self.assertTrue(entry.public_preflight(game_root='synthetic', registration_commit='a' * 40))
            bind.assert_not_called()
            claim.assert_not_called()
        with patch.object(entry, '_context', return_value=ctx), \
             patch.object(entry.os.path, 'lexists', return_value=True), self.assertRaises(ValueError):
            entry.public_preflight(game_root='synthetic', registration_commit='a' * 40)

    def test_result_reader_rechecks_receipts_and_rejects_comparison_swap(self):
        stage = Path('synthetic-stage')
        files, bundles = {}, {}
        plans = [dict(game=game) for game in entry.GAMES]
        rows = [dict(game=game, replay_sha256='synthetic-' + game) for game in entry.GAMES]
        comparisons = [dict(seed_alias=alias, side=side) for alias in ('S02', 'S03') for side in (0, 1)]
        def raw(value):
            return json.dumps(value, sort_keys=True).encode()
        def sha(value):
            return hashlib.sha256(value).hexdigest()
        for plan in plans:
            game = plan['game']
            bundle = {name: raw({'synthetic': name}) for name in ('row', 'audit', 'complete', 'storage',
                                                                   'metrics', 'metrics_receipt')}
            bundles[game] = bundle
            supervision_raw = raw(dict(schema='task-pair-dev-job-supervision-v1', game=game,
                                       retry_allowed=False, supervision={'passed': True}))
            input_raw = raw({'synthetic': game})
            accepted = dict(schema='task-pair-dev-job-acceptance-v1', game=game, passed=True,
                            retry_allowed=False, supervision_sha256=sha(supervision_raw),
                            job_input_sha256=sha(input_raw),
                            receipt_sha256={name: sha(value) for name, value in bundle.items()})
            files[stage / 'job-results' / (game + '.supervision.local.json')] = supervision_raw
            files[stage / 'job-results' / (game + '.acceptance.local.json')] = raw(accepted)
            files[stage / 'job-inputs' / (game + '.json')] = input_raw
        def comparison_path(root, game, candidate):
            return root / 'job-results' / (game + '-to-' + candidate + '.comparison.local.json')
        for item in comparisons:
            game = f"dev-{item['seed_alias']}-market-{item['side']}-baseline"
            files[comparison_path(stage, game, 'control')] = raw({'comparison': item})
        dmod = SimpleNamespace(_read=lambda _root, path, _limit: files[path],
            _hash=lambda _root, path, _limit: sha(files[path]), _comparison_file=comparison_path,
            _replay=lambda _root, _path, digest: {'synthetic_digest': digest},
            _write_new=lambda _root, path, value: files.__setitem__(path, raw(value)))
        acceptance = SimpleNamespace(MAX_RECEIPT_BYTES=100000, _json=lambda data, _limit: json.loads(data),
            accept=lambda _panel, plan, _bundle, _supervision, _verify:
                {'game': plan['game'], 'replay_sha256': 'synthetic-' + plan['game']})
        driver = SimpleNamespace(read_bundle=lambda plan: bundles[plan['game']], verify_files=lambda *_args: True,
                                 preflight_comparison=lambda *_args: True)
        projected = {'sanitized': True, 'pairs': [dict(seed_alias=alias, side=side)
                     for alias in ('S01', 'S04') for side in (0, 1)]}
        ctx = dict(stage=stage, plans=plans, manifest=SimpleNamespace(panel=SimpleNamespace(seeds=(1, 2))),
                   once_auth=once_auth, projection=SimpleNamespace(project=lambda *_args, **_kw: projected),
                   request_prefix=SimpleNamespace(compare=lambda *_args, **_kw: {'status': 'none'}),
                   time_check=lambda: True,
                   commit='a' * 40, attempt_raw=b'attempt', manifest_raw=b'manifest',
                   limits={'total_output_bytes': 1000000, 'disk_free_floor_bytes': 1},
                   modules={'driver': dmod, 'acceptance': acceptance,
                            'child': SimpleNamespace(worker=SimpleNamespace(MAX_JOB_BYTES=16384))})
        for game in entry.GAMES:
            for kind in ('permit', 'started', 'consumed'):
                name, marker = entry._marker(ctx, game, kind)
                files[stage / name] = entry._json(marker)
        dmod.MAX_SUMMARY = 1000000
        dmod._owned_size = lambda _path: 0
        payload = dict(attempted=8, official_replay_passes=8, training=False, promotion=False,
                       independent_improvement=False, rows=rows, comparisons=comparisons, paired={})
        with patch.object(entry, '_check', return_value=True), \
             patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=1000000)):
            self.assertTrue(entry._save_results(ctx, driver, payload))
            self.assertEqual(json.loads(files[stage / 'projected-summary.local.json']), projected)
            wrong = comparison_path(stage, 'dev-S02-market-0-baseline', 'control')
            files[wrong] = raw({'comparison': comparisons[1]})
            with self.assertRaises(ValueError):
                entry._save_results(ctx, driver, payload)
            files[wrong] = raw({'comparison': comparisons[0]})
            marker = stage / entry._marker(ctx, entry.GAMES[0], 'consumed')[0]
            original = files[marker]
            files[marker] = b'{}'
            with self.assertRaises(ValueError):
                entry._save_results(ctx, driver, payload)
            files[marker] = original
            with patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
                with self.assertRaises(ValueError):
                    entry._save_results(ctx, driver, payload)


class AuthTests(unittest.TestCase):
    def test_registered_scope_cannot_change_types_roles_or_evaluation(self):
        identity = SimpleNamespace(ENGINE_COMMIT='engine', ENGINE_SHA='rules', RESULT_COMMIT='training',
                                   HISTORICAL_RESULT='history')
        reg = dict(schema=auth.SCHEMA, stage_name=auth.STAGE_NAME, panel=copy.deepcopy(auth.PANEL),
                   evaluation=copy.deepcopy(auth.EVALUATION), engine_commit='engine', engine_sha256='rules',
                   previous_training_result_commit='training', historical_panel_result_commit='history',
                   public_sources={'required': 'x'})
        auth._panel(reg, identity, once_auth, {'required'})
        changes = [
            ('panel', 'games', True), ('panel', 'actual_seed_roles', ['S01', 'S03']),
            ('panel', 'opponents', ['teacher']), ('evaluation', 'retries', True),
            ('evaluation', 'official_games', 4), ('evaluation', 'independent_evaluation', True),
        ]
        for section, key, value in changes:
            changed = copy.deepcopy(reg)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                auth._panel(changed, identity, once_auth, {'required'})


if __name__ == '__main__':
    unittest.main()
