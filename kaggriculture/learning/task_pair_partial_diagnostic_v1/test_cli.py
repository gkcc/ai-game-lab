"""Synthetic registration, memory-gate and ledger checks; no real IO."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import stat
from types import SimpleNamespace
import unittest
from unittest import mock

from . import cli
from learning.task_pair_dev_campaign_v1 import acceptance
from learning.task_pair_dev_campaign_v1.test_acceptance import fixture as accepted_fixture, raw, supervision


def registration():
    return dict(schema=cli.SCHEMA, original_commit=cli.ORIGINAL_COMMIT, scope=deepcopy(cli.SCOPE),
                limits=dict(cli.LIMITS), failure_commit='a' * 40,
                failure_certificate='review/evidence/synthetic/certificate.json',
                failure_certificate_sha256='b' * 64,
                public_sources={key: 'c' * 64 for key in cli._REQUIRED})


def evidence():
    return dict(schema=cli.LEDGER_SCHEMA, registration_commit=cli.ORIGINAL_COMMIT,
                stage_files={f'file-{i:02}.bin': dict(bytes=1, sha256='a' * 64) for i in range(79)},
                attempt=dict(bytes=1, sha256='b' * 64), failure=dict(bytes=1, sha256='c' * 64),
                observed_outer_exit_code=1, observed_outer_exit_source='operator tool session',
                recorded_after_exit=True, retry_allowed=False)


def accepted_pair_fixture():
    """Construct two linked bundles entirely in memory using the real validator."""
    manifest, _, template = accepted_fixture()
    bundles, files, job_identity = {}, {}, 'f' * 64
    attempt, manifest_raw = b'synthetic-attempt', b'synthetic-manifest'
    for arm in ('baseline', 'control'):
        plan = next(p for p in manifest.panel.plans() if p['game'] == f'dev-S02-market-0-{arm}')
        game = plan['game']
        values = {name: json.loads(data) for name, data in template.items()}
        values['row'].update(plan)
        values['row']['task_checkpoint_sha256'] = manifest.panel.task_model(plan)[1]
        values['audit']['game'] = game
        for name in ('primary_model_counts', 'live_model_counts'):
            values['audit'][name]['crop_forwards'] = 1
        values['complete']['game'] = game
        values['storage']['game'] = game
        metric = values['metrics']
        metric.update(game=game, seed=plan['seed'], seed_alias=plan['seed_alias'],
                      opponent=plan['opponent'], side=plan['side'], arm=arm,
                      task_checkpoint_sha256=manifest.panel.task_model(plan)[1])
        values['metrics_receipt']['game'] = game
        bundle = {name: raw(value) for name, value in values.items()}
        values['complete']['row_sha256'] = cli._sha(bundle['row'])
        values['complete']['certificate_sha256'] = cli._sha(bundle['audit'])
        bundle['complete'] = raw(values['complete'])
        values['metrics_receipt'].update(metrics_bytes=len(bundle['metrics']),
                                         metrics_sha256=cli._sha(bundle['metrics']))
        bundle['metrics_receipt'] = raw(values['metrics_receipt'])
        bundles[game] = bundle
        for kind in ('permit', 'started', 'consumed'):
            files[game + '.' + kind + '.local.json'] = cli._encoded(
                cli._marker(game, kind, attempt, manifest_raw))
        supervision_raw = raw(dict(schema='task-pair-dev-job-supervision-v1', game=game,
                                   retry_allowed=False, supervision=supervision()))
        files[game + '.supervision.local.json'] = supervision_raw
        files[game + '.acceptance.local.json'] = raw(dict(
            schema='task-pair-dev-job-acceptance-v1', game=game, passed=True, retry_allowed=False,
            supervision_sha256=cli._sha(supervision_raw), job_input_sha256=job_identity,
            receipt_sha256={name: cli._sha(data) for name, data in bundle.items()}))
    class Driver:
        def __init__(self, **_kwargs): pass
        def read_bundle(self, plan): return bundles[plan['game']]
        def verify_files(self, _plan, _row): return True
    dmod = SimpleNamespace(LocalDriver=Driver, _read=lambda _stage, path, _limit: files[path.name],
                           _hash=lambda *_args: job_identity)
    child = SimpleNamespace(worker=SimpleNamespace(MAX_JOB_BYTES=16384))
    ctx = dict(root=Path('synthetic-root'), modules=dict(driver=dmod, acceptance=acceptance, child=child))
    return ctx, manifest, attempt, manifest_raw, files


class CliTests(unittest.TestCase):
    def test_system_exit_never_becomes_a_successful_diagnostic(self):
        args = ['--game-root', 'synthetic', '--registration-commit', 'a' * 40,
                '--evidence-file', 'synthetic', '--evidence-sha256', 'b' * 64]
        for code in (0, None, 'SYNTHETIC_PRIVATE_VALUE'):
            with self.subTest(code=code), mock.patch.object(cli, 'run', side_effect=SystemExit(code)):
                with self.assertRaises(SystemExit) as stopped:
                    cli.main(args)
                self.assertNotIn(stopped.exception.code, (0, None))
                self.assertNotIn('SYNTHETIC_PRIVATE_VALUE', str(stopped.exception))

    def test_success_receipt_binds_exact_result_and_return_binds_receipt(self):
        result = b'{"synthetic":true}\n'
        receipt, returned = cli._success_receipts(result, 'a' * 40, 'b' * 64)
        self.assertEqual(json.loads(receipt)['result_sha256'], hashlib.sha256(result).hexdigest())
        self.assertEqual(json.loads(returned)['receipt_sha256'], hashlib.sha256(receipt).hexdigest())
        self.assertNotEqual(json.loads(receipt)['result_sha256'], hashlib.sha256(result + b' ').hexdigest())

    def test_two_original_receipt_chains_run_real_acceptance_without_games_or_io(self):
        ctx, manifest, attempt, manifest_raw, files = accepted_pair_fixture()
        stage = Path('synthetic-root/stage')
        def run():
            return cli._accepted_pair(ctx, stage, manifest, attempt, manifest_raw,
                                      check=lambda: True, guard=lambda: True)
        _, plans, rows = run()
        self.assertEqual(set(plans), {'baseline', 'control'})
        self.assertEqual(set(rows), {'baseline', 'control'})
        for suffix in ('permit', 'supervision', 'acceptance'):
            key = f'dev-S02-market-0-control.{suffix}.local.json'
            original = files[key]
            files[key] = (raw(dict(json.loads(original), passed=False))
                          if suffix == 'acceptance' else original + b' ')
            with self.subTest(receipt=suffix), self.assertRaises(ValueError):
                run()
            files[key] = original

    def test_registration_fixed_scope_and_limits(self):
        cli._registration(registration())
        for edit in ('games', 'transition', 'side', 'metric_timing', 'memory', 'missing_source'):
            value = registration()
            if edit == 'games': value['scope']['new_games'] = 1
            if edit == 'transition': value['scope']['new_official_transitions'] = 1
            if edit == 'side': value['scope']['side'] = True
            if edit == 'metric_timing': value['scope']['metrics_fields_previously_read'] = False
            if edit == 'memory': value['limits']['during_available_bytes'] -= 1
            if edit == 'missing_source': value['public_sources'].pop(next(iter(cli._REQUIRED)))
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                cli._registration(value)

    def test_ledger_requires_79_unique_bounded_post_exit_files(self):
        cli._ledger(evidence())
        for edit in ('missing', 'exit', 'exit_source', 'retry', 'timestamp', 'size', 'digest', 'escape', 'casefold'):
            value = evidence()
            if edit == 'missing': value['stage_files'].pop('file-00.bin')
            if edit == 'exit': value['observed_outer_exit_code'] = 0
            if edit == 'exit_source': value['observed_outer_exit_source'] = 'unverified source'
            if edit == 'retry': value['retry_allowed'] = True
            if edit == 'timestamp': value['recorded_after_exit'] = False
            if edit == 'size': value['stage_files']['file-00.bin']['bytes'] = -1
            if edit == 'digest': value['stage_files']['file-00.bin']['sha256'] = 'not-a-digest'
            if edit == 'escape': value['stage_files']['../secret'] = value['stage_files'].pop('file-00.bin')
            if edit == 'casefold': value['stage_files']['FILE-01.bin'] = value['stage_files'].pop('file-00.bin')
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                cli._ledger(value)

    def test_memory_and_monotonic_time_use_separate_initial_and_during_thresholds(self):
        available = [cli.LIMITS['initial_available_bytes']]
        psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=available[0]))
        values = iter((1., 2., 3., 4.))
        guard = cli._Guard(psutil, 0., clock=lambda: next(values))
        self.assertTrue(guard.sample(initial=True))
        available[0] = cli.LIMITS['during_available_bytes']
        self.assertTrue(guard.sample())
        available[0] -= 1
        with self.assertRaises(ValueError): guard.sample()
        guard = cli._Guard(psutil, 5., clock=lambda: 4.)
        with self.assertRaises(ValueError): guard.sample()
        available[0] = cli.LIMITS['initial_available_bytes'] - 1
        guard = cli._Guard(psutil, 0., clock=lambda: 1.)
        with self.assertRaises(ValueError): guard.sample(initial=True)

    def test_stage_file_set_and_81_hash_checks_use_only_given_paths(self):
        value = evidence()
        root, stage = Path('synthetic-root'), Path('synthetic-root/stage')
        driver = SimpleNamespace(_no_links=mock.Mock())
        info = SimpleNamespace(st_mode=stat.S_IFREG, st_size=1)
        files = list(value['stage_files'])
        with mock.patch.object(cli.os, 'walk', return_value=[(str(stage), [], files)]), \
             mock.patch.object(Path, 'lstat', return_value=info), \
             mock.patch.object(cli, '_hash_file') as check:
            self.assertTrue(cli._stage_snapshot(root, stage, value, driver, lambda: True))
            self.assertEqual(check.call_count, 81)
        with mock.patch.object(cli.os, 'walk', return_value=[(str(stage), [], files[:-1])]), \
             mock.patch.object(Path, 'lstat', return_value=info):
            with self.assertRaises(ValueError):
                cli._stage_snapshot(root, stage, value, driver, lambda: True)

    def test_hash_file_checks_bytes_and_digest_without_writing(self):
        raw = b'synthetic-bytes'
        meta = dict(bytes=len(raw), sha256=cli._sha(raw))
        root, path = Path('synthetic-root'), Path('synthetic-root/file')
        info = SimpleNamespace(st_mode=stat.S_IFREG, st_size=len(raw))
        driver = SimpleNamespace(_no_links=mock.Mock())
        with mock.patch.object(Path, 'lstat', return_value=info), \
             mock.patch.object(Path, 'open', side_effect=lambda *args, **kwargs: io.BytesIO(raw)):
            cli._hash_file(root, path, meta, driver, lambda: True)
            with self.assertRaises(ValueError):
                cli._hash_file(root, path, dict(meta, sha256='a' * 64), driver, lambda: True)

    def test_output_budget_covers_the_entire_new_stage(self):
        root, path = mock.Mock(), mock.MagicMock()
        info = SimpleNamespace(st_mode=stat.S_IFREG, st_size=7)
        item = SimpleNamespace(lstat=lambda: info)
        root.iterdir.return_value = iter([item])
        stream = mock.MagicMock()
        stream.write.side_effect = lambda data: len(data)
        path.open.return_value.__enter__.return_value = stream
        with mock.patch.dict(cli.LIMITS, output_bytes=10, disk_free_floor_bytes=4), \
             mock.patch.object(cli.shutil, 'disk_usage', return_value=SimpleNamespace(free=100)), \
             mock.patch.object(cli.os, 'fsync'), mock.patch.object(cli, '_read', return_value=b'123'):
            with self.assertRaises(ValueError): cli._write_new(path, b'1234', root)
            root.iterdir.return_value = iter([item])
            cli._write_new(path, b'123', root)

    def test_marker_and_json_are_strict_without_outputting_inputs(self):
        marker = cli._marker('dev-S02-market-0-baseline', 'permit', b'synthetic-attempt', b'synthetic-manifest')
        self.assertEqual((marker['registration_commit'], marker['retry_allowed']), (cli.ORIGINAL_COMMIT, False))
        self.assertIn('manifest_sha256', marker)
        self.assertNotIn('manifest_sha256', cli._marker('dev-S02-market-0-baseline', 'started', b'a', b'b'))
        with self.assertRaises(ValueError): cli._json(b'{"a":1,"a":2}', 100)
        with self.assertRaises(ValueError): cli._json(b'{"a":NaN}', 100)
        with self.assertRaises(ValueError): cli._safe_relative('..\\secret')
        with self.assertRaises(ValueError): cli._safe_relative('NUL.json')


if __name__ == '__main__':
    unittest.main()
