"""Entry safety contracts using in-memory substitutes, never a real game."""
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from learning.task_anchor_once_v1 import storage
from learning.task_pair_dev_campaign_v1.manifest import build_manifest
from . import entry
from .test_runner import manifest_and_plans, ROOT, STAGE


class _MemoryOutput(io.BytesIO):
    def close(self):
        pass

    def fileno(self):
        return 17


def fake_context():
    manifest, plans = manifest_and_plans()
    marker = ROOT / 'artifacts' / (STAGE.name + '.attempt.local.json')
    driver = SimpleNamespace(_no_links=mock.Mock(), _read=mock.Mock(), _hash=mock.Mock(),
                             _owned_size=mock.Mock(return_value=100),
                             MAX_MODEL=16 * 1024**2, MAX_SOURCE=4 * 1024**2)
    ctx = dict(game_root=ROOT, stage=STAGE, commit='a' * 40, manifest=manifest, plans=plans,
               attempt_path=marker, attempt_raw=b'synthetic attempt', manifest_raw=b'synthetic manifest',
               limits={'total_output_bytes': 2 * 1024**2, 'disk_free_floor_bytes': 1024**2},
               modules={'driver': driver, 'child': SimpleNamespace(run_one=mock.Mock()),
                        'acceptance': SimpleNamespace(supervision_ok=mock.Mock(return_value=True))},
               identity=SimpleNamespace(_json=lambda value: json.loads(value)))
    return ctx, driver


class SnapshotTests(unittest.TestCase):
    def test_stub_checks_all_four_blobs_before_executing_entry(self):
        repo, commit = Path('synthetic-toy-repo'), 'a' * 40
        sources = {
            entry.TOY + 'entry.py': b'import sys; sys._toy_snapshot_probe=True\n',
            entry.CAMPAIGN + 'identity.py': b'IDENTITY=1\n',
            entry.CAMPAIGN + 'bootstrap.py': b'BOOTSTRAP=1\n',
            entry.TOY + 'toy_auth.py': b'AUTH=1\n',
        }
        calls = []

        def read(path, mode):
            self.assertEqual(mode, 'rb')
            return io.BytesIO(sources[path.relative_to(repo).as_posix()])

        def git(argv, **_kwargs):
            rel = argv[-1].split(':', 1)[1]
            calls.append((argv[3], rel))
            return SimpleNamespace(stdout=(str(len(sources[rel])).encode() if argv[3] == 'cat-file'
                                           else sources[rel]))

        if hasattr(sys, '_toy_snapshot_probe'):
            delattr(sys, '_toy_snapshot_probe')
        try:
            with mock.patch.object(Path, 'open', autospec=True, side_effect=read), \
                 mock.patch.object(subprocess, 'run', side_effect=git), \
                 mock.patch.object(sys, 'argv', ['-c', str(repo), commit, '--run']):
                exec(entry.SNAPSHOT_STUB, {})
            self.assertTrue(sys._toy_snapshot_probe)
            self.assertEqual(len(calls), 8)
            self.assertEqual({rel for _, rel in calls}, set(sources))
        finally:
            if hasattr(sys, '_toy_snapshot_probe'):
                delattr(sys, '_toy_snapshot_probe')

    def test_changed_blob_stops_before_entry_execution(self):
        repo, commit = Path('synthetic-toy-repo'), 'a' * 40
        def read(_path, _mode):
            return io.BytesIO(b'unchanged')
        def git(argv, **_kwargs):
            return SimpleNamespace(stdout=(b'9' if argv[3] == 'cat-file' else b'changed!!'))
        with mock.patch.object(Path, 'open', autospec=True, side_effect=read), \
             mock.patch.object(subprocess, 'run', side_effect=git), \
             mock.patch.object(sys, 'argv', ['-c', str(repo), commit, '--run']):
            with self.assertRaises(SystemExit):
                exec(entry.SNAPSHOT_STUB, {})

    def test_snapshots_must_be_supplied_in_memory(self):
        with mock.patch.dict(entry.__dict__, {'_AUTH_IDENTITY_BYTES': None,
                                             '_AUTH_BOOTSTRAP_BYTES': b'B=1',
                                             '_AUTH_TOY_AUTH_BYTES': b'C=1'}):
            with self.assertRaises(ValueError): entry._snapshot_modules()
        with mock.patch.dict(entry.__dict__, {'_AUTH_IDENTITY_BYTES': b'A=1',
                                             '_AUTH_BOOTSTRAP_BYTES': b'B=2',
                                             '_AUTH_TOY_AUTH_BYTES': b'C=3'}):
            a, b, c = entry._snapshot_modules()
        self.assertEqual((a.A, b.B, c.C), (1, 2, 3))


class EntryTests(unittest.TestCase):
    def test_context_uses_authenticated_sources_and_independent_stage(self):
        blobs = {name: ('toy-' + name).encode() for name in ('baseline', 'control', 'auxiliary', 'crop')}
        sha = {name: hashlib.sha256(raw).hexdigest() for name, raw in blobs.items()}
        market, other, worker_source, registration = b'market-source', b'other-source', b'worker-source', b'{}'
        reg = dict(toy_model_sha256=sha, limits={'synthetic': True}, public_sources={
            'kaggriculture/bots/heuristic_v1_market.py': hashlib.sha256(market).hexdigest(),
            'kaggriculture/bots/heuristic_v1.py': hashlib.sha256(other).hexdigest()})
        crop = dict(features=['synthetic'], schema_version='crop', teacher_sha256='t', schema_sha256='c')
        task = dict(schema_version='task')
        sources = {entry.PUBLIC_ROOT / entry.CROP_SCHEMA: json.dumps(crop).encode(),
                   entry.PUBLIC_ROOT / entry.TASK_SCHEMA: json.dumps(task).encode(),
                   ROOT / 'bots/heuristic_v1_market.py': market,
                   ROOT / 'bots/heuristic_v1.py': other,
                   ROOT / 'learning/task_anchor_once_v1/games.py': worker_source,
                   entry.PUBLIC_ROOT / 'synthetic-registration.json': registration}
        identity = SimpleNamespace(_json=lambda raw: json.loads(raw),
            _read=mock.Mock(side_effect=lambda path: sources[path]),
            _git=mock.Mock(return_value=registration))
        modules = {'manifest': SimpleNamespace(build_manifest=build_manifest)}
        order = []
        bootstrap = SimpleNamespace(_absolute=lambda _path: ROOT,
            _runtime=mock.Mock(side_effect=lambda *_: order.append('runtime') or modules))
        auth = SimpleNamespace(STAGE_NAME='task-pair-toy-smoke-v1-20261030',
            REGISTRATION='synthetic-registration.json',
            verify_public=mock.Mock(side_effect=lambda *_a, **_k: order.append('auth') or
                (reg, {}, {'runtime_files': {'learning/task_anchor_once_v1/games.py':
                                              hashlib.sha256(worker_source).hexdigest()}})))
        archives = dict(trained=False, crop=dict(data=blobs['crop'], sha256=sha['crop']),
                        tasks={role: dict(data=blobs[role], sha256=sha[role])
                               for role in ('baseline', 'control', 'auxiliary')})
        weights = SimpleNamespace(CROPS=('A', 'B'), build=mock.Mock(return_value=archives))
        runner = SimpleNamespace(limits_snapshot=mock.Mock(return_value={'synthetic': True}))
        with mock.patch.object(entry, '_snapshot_modules', return_value=(identity, bootstrap, auth)), \
             mock.patch.object(entry, '_game_root', return_value=ROOT), \
             mock.patch.object(entry, '_check_toy_package', side_effect=lambda *_: order.append('shadow_check')), \
             mock.patch.object(entry, '_load_toy', side_effect=lambda name: {'toy_weights': weights, 'runner': runner}[name]):
            ctx = entry._build_context(ROOT, 'a' * 40)
            self.assertEqual(order, ['auth', 'shadow_check', 'runtime'])
            self.assertEqual(ctx['stage'], ROOT / 'artifacts' / auth.STAGE_NAME)
            self.assertEqual(ctx['attempt_path'], ROOT / 'artifacts' / (auth.STAGE_NAME + '.attempt.local.json'))
            self.assertEqual([plan['game'] for plan in ctx['plans']], list(entry._ALLOWED))
            self.assertEqual(ctx['manifest'].panel.seeds, (11017, 22029))
            archives['tasks']['baseline']['data'] = b'changed'
            with self.assertRaisesRegex(ValueError, 'checkpoint identities differ'):
                entry._build_context(ROOT, 'a' * 40)

    def test_game_root_is_bound_to_the_publication_workspace(self):
        expected = entry.PUBLIC_ROOT.parent.parent / 'kaggriculture'
        bootstrap = SimpleNamespace(_absolute=lambda value: value)
        with mock.patch.object(Path, 'resolve', autospec=True, side_effect=lambda path, **_: path):
            self.assertEqual(entry._game_root(bootstrap, expected), expected)
            with self.assertRaises(ValueError):
                entry._game_root(bootstrap, expected.parent / 'another-game')

    def test_invalid_modes_and_unregistered_game_never_reach_context(self):
        with mock.patch.object(entry, '_build_context') as build:
            with self.assertRaises(ValueError):
                entry.execute(game_root=ROOT, registration_commit='a' * 40, game='wrong')
            build.assert_not_called()
        with mock.patch.object(entry, 'execute') as execute:
            for args in ([], ['--game-root', str(ROOT), '--registration-commit', 'a' * 40],
                         ['--game-root', str(ROOT), '--registration-commit', 'a' * 40,
                          '--run', '--game', entry._ALLOWED[0]]):
                with self.subTest(args=len(args)), self.assertRaises(ValueError): entry.main(args)
            execute.assert_not_called()

    def test_attempt_marker_rejects_existing_stage_or_prior_attempt(self):
        ctx, driver = fake_context()
        for existing in (ctx['stage'], ctx['attempt_path']):
            with self.subTest(kind=existing.name), \
                 mock.patch.object(entry.os.path, 'lexists', side_effect=lambda path: path == existing), \
                 mock.patch.object(Path, 'open', autospec=True) as opened:
                with self.assertRaises(ValueError): entry._claim_attempt(ctx)
                opened.assert_not_called()
        self.assertTrue(driver._no_links.called)

    def test_attempt_marker_is_exclusive_and_read_back_before_runner(self):
        ctx, _ = fake_context()
        output = _MemoryOutput()
        with mock.patch.object(entry.os.path, 'lexists', return_value=False), \
             mock.patch.object(Path, 'open', autospec=True, return_value=output) as opened, \
             mock.patch.object(entry.os, 'fsync') as fsync, \
             mock.patch.object(entry, '_attempt_check', return_value=True):
            entry._claim_attempt(ctx)
        opened.assert_called_once_with(ctx['attempt_path'], 'xb')
        self.assertEqual(output.getvalue(), ctx['attempt_raw'])
        fsync.assert_called_once_with(17)

    def test_local_toy_shadow_fails_before_import_or_public_read(self):
        root = Path('synthetic-game')
        shadow = root / 'learning' / 'task_pair_toy_smoke_v1'
        with mock.patch.object(entry.os.path, 'lexists', side_effect=lambda path: path == shadow), \
             mock.patch.object(Path, 'open', autospec=True) as opened:
            with self.assertRaises(ValueError): entry._check_toy_package(root, {})
            opened.assert_not_called()

    def test_already_imported_toy_package_is_rejected(self):
        raw = b'synthetic source'
        sources = {entry.TOY + name + '.py': hashlib.sha256(raw).hexdigest()
                   for name in ('__init__', 'toy_auth', 'toy_weights', 'runner', 'entry')}
        with mock.patch.object(entry.os.path, 'lexists', return_value=False), \
             mock.patch.object(Path, 'is_dir', autospec=True, return_value=True), \
             mock.patch.object(Path, 'resolve', autospec=True, side_effect=lambda path, **_: path), \
             mock.patch.object(Path, 'open', autospec=True, side_effect=lambda *_: io.BytesIO(raw)):
            with self.assertRaisesRegex(ValueError, 'imported early'):
                entry._check_toy_package(Path('synthetic-game'), {'public_sources': sources})

    def test_child_is_not_started_if_stage_validation_fails(self):
        ctx, _ = fake_context()
        with mock.patch.object(entry, '_build_context', return_value=ctx), \
             mock.patch.object(entry, '_git_check', return_value=True), \
             mock.patch.object(entry, '_attempt_check', return_value=True), \
             mock.patch.object(entry, '_child_ready', side_effect=ValueError('synthetic stage failure')):
            with self.assertRaises(ValueError):
                entry.execute(game_root=ROOT, registration_commit='a' * 40, game=entry._ALLOWED[0])
        ctx['modules']['child'].run_one.assert_not_called()

    def test_parent_marks_attempt_before_runner_and_returns_only_after_marker(self):
        ctx, driver = fake_context()
        local = SimpleNamespace(stage=STAGE, claimed=True, launch=mock.Mock())
        ctx['limits'] = {'synthetic': True}
        ctx['source'] = b'synthetic worker'
        order = []
        driver.LocalDriver = mock.Mock(side_effect=lambda **_kwargs: order.append('driver') or local)
        result = dict(passed=True, scope='synthetic_toy', games=2, official_replay_passes=2,
                      comparisons=1, promotion=False)
        def run(**kwargs):
            order.append('runner')
            self.assertTrue(kwargs['check']())
            self.assertTrue(kwargs['prepare_inputs'](ctx['manifest'], local, 10))
            self.assertTrue(kwargs['check']())
            self.assertTrue(kwargs['save_result']({'synthetic': True}))
            return result
        ctx['runner'] = SimpleNamespace(run=run)
        with mock.patch.object(entry, '_build_context', return_value=ctx), \
             mock.patch.object(entry, '_claim_attempt', side_effect=lambda _ctx: order.append('attempt')), \
             mock.patch.object(entry, '_prepare', return_value=True), \
             mock.patch.object(entry, '_save', return_value=True), \
             mock.patch.object(entry, '_check', return_value=True) as check, \
             mock.patch.object(entry, '_mark_returned', side_effect=lambda *_: order.append('returned')):
            returned = entry.execute(game_root=ROOT, registration_commit='a' * 40)
        self.assertEqual(order, ['attempt', 'driver', 'runner', 'returned'])
        self.assertEqual(check.call_args_list, [mock.call(ctx, False), mock.call(ctx, True)])
        self.assertTrue(returned['runner_completed'])
        self.assertIs(returned['observed_outer_exit'], False)
        self.assertNotIn('passed', returned)

    def test_child_receives_only_registered_plan_after_all_guards(self):
        ctx, _ = fake_context()
        ctx['source'] = b'synthetic worker'
        with mock.patch.object(entry, '_build_context', return_value=ctx), \
             mock.patch.object(entry, '_git_check', return_value=True), \
             mock.patch.object(entry, '_attempt_check', return_value=True), \
             mock.patch.object(entry, '_child_ready') as ready, \
             mock.patch.object(entry, '_consume_game') as consumed:
            self.assertIsNone(entry.execute(game_root=ROOT, registration_commit='a' * 40,
                                            game=entry._ALLOWED[0]))
        ready.assert_called_once_with(ctx, entry._ALLOWED[0])
        consumed.assert_called_once_with(ctx, entry._ALLOWED[0])
        call = ctx['modules']['child'].run_one.call_args.kwargs
        self.assertEqual(call['plan'], ctx['plans'][0])
        self.assertEqual(call['worker_source'], b'synthetic worker')
        self.assertEqual(call['stage'], ctx['stage'])

    def test_child_without_permit_or_after_consumption_never_starts(self):
        ctx, driver = fake_context()
        ctx['source'] = b'synthetic worker'
        game = entry._ALLOWED[0]
        consumed, _ = entry._game_marker(ctx, game, 'consumed')
        permit, raw = entry._game_marker(ctx, game, 'permit')
        with mock.patch.object(entry, '_build_context', return_value=ctx), \
             mock.patch.object(entry, '_git_check', return_value=True), \
             mock.patch.object(entry, '_attempt_check', return_value=True), \
             mock.patch.object(entry, '_child_ready'), \
             mock.patch.object(entry, '_write_game_marker') as write:
            driver._read.side_effect = FileNotFoundError('synthetic missing permit')
            with mock.patch.object(entry.os.path, 'lexists', return_value=False):
                with self.assertRaises(FileNotFoundError):
                    entry.execute(game_root=ROOT, registration_commit='a' * 40, game=game)
            driver._read.side_effect = None
            driver._read.return_value = raw
            with mock.patch.object(entry.os.path, 'lexists', side_effect=lambda path: path == consumed):
                with self.assertRaises(ValueError):
                    entry.execute(game_root=ROOT, registration_commit='a' * 40, game=game)
            write.assert_not_called()
        ctx['modules']['child'].run_one.assert_not_called()

    def test_parent_authorizes_only_registered_plan_once_before_launch(self):
        ctx, _ = fake_context()
        game = entry._ALLOWED[0]
        permit, _ = entry._game_marker(ctx, game, 'permit')
        with mock.patch.object(entry, '_check', return_value=True), \
             mock.patch.object(entry.os.path, 'lexists', return_value=False), \
             mock.patch.object(entry, '_write_game_marker') as write:
            entry._authorize_game(ctx, ctx['plans'][0])
            write.assert_called_once_with(ctx, game, 'permit')
            invalid = dict(ctx['plans'][0], side=1)
            with self.assertRaises(ValueError): entry._authorize_game(ctx, invalid)
        with mock.patch.object(entry, '_check', return_value=True), \
             mock.patch.object(entry.os.path, 'lexists', side_effect=lambda path: path == permit), \
             mock.patch.object(entry, '_write_game_marker') as write:
            with self.assertRaises(ValueError): entry._authorize_game(ctx, ctx['plans'][0])
            write.assert_not_called()

    def test_permit_is_consumed_exclusively_before_worker(self):
        ctx, driver = fake_context()
        game = entry._ALLOWED[0]
        _, raw = entry._game_marker(ctx, game, 'permit')
        driver._read.return_value = raw
        with mock.patch.object(entry.os.path, 'lexists', return_value=False), \
             mock.patch.object(entry, '_write_game_marker') as write:
            entry._consume_game(ctx, game)
            write.assert_called_once_with(ctx, game, 'consumed')

    def test_game_marker_uses_owned_bounded_exclusive_write(self):
        ctx, driver = fake_context()
        game = entry._ALLOWED[0]
        path, raw = entry._game_marker(ctx, game, 'permit')
        written = {}
        driver._read.side_effect = lambda _root, candidate, _max: written[candidate]
        with mock.patch.object(storage, 'owned_writes', return_value=nullcontext()) as budget, \
             mock.patch.object(entry, '_ensure_room') as room, \
             mock.patch.object(entry, '_write_exact', side_effect=lambda candidate, data: written.update({candidate: data})):
            entry._write_game_marker(ctx, game, 'permit')
        room.assert_called_once_with(ctx, len(raw))
        self.assertEqual(written[path], raw)
        self.assertEqual(budget.call_args.kwargs['allowed_paths'], ['job-results/' + path.name])

    def test_first_launch_failure_does_not_authorize_a_second_game(self):
        ctx, driver = fake_context()
        ctx['limits'] = {'synthetic': True}
        ctx['source'] = b'synthetic worker'
        calls = []
        def launch(plan, _limits):
            calls.append(('launch', plan['game']))
            raise RuntimeError('synthetic first launch failure')
        local = SimpleNamespace(stage=STAGE, claimed=True, launch=launch)
        driver.LocalDriver = mock.Mock(return_value=local)
        def run(**kwargs):
            kwargs['driver'].launch(ctx['plans'][0], {})
            kwargs['driver'].launch(ctx['plans'][1], {})
        ctx['runner'] = SimpleNamespace(run=run)
        with mock.patch.object(entry, '_build_context', return_value=ctx), \
             mock.patch.object(entry, '_claim_attempt'), \
             mock.patch.object(entry, '_authorize_game', side_effect=lambda _ctx, plan: calls.append(('permit', plan['game']))), \
             mock.patch.object(entry, '_mark_returned') as marked:
            with self.assertRaises(RuntimeError):
                entry.execute(game_root=ROOT, registration_commit='a' * 40)
        self.assertEqual(calls, [('permit', entry._ALLOWED[0]), ('launch', entry._ALLOWED[0])])
        marked.assert_not_called()

    def test_git_or_registration_change_fails_closed(self):
        ctx, _ = fake_context()
        ctx['auth'] = SimpleNamespace(REGISTRATION='synthetic-registration')
        ctx['reg_raw'] = b'registered'
        ctx['identity']._git = mock.Mock(side_effect=[b'other', b'', b''])
        ctx['identity']._read = mock.Mock(return_value=b'registered')
        self.assertFalse(entry._git_check(ctx))
        self.assertEqual(ctx['identity']._git.call_count, 1)
        ctx['identity']._git = mock.Mock(side_effect=[('a' * 40).encode(), b'dirty'])
        self.assertFalse(entry._git_check(ctx))
        ctx['identity']._read.assert_not_called()
        ctx['identity']._git = mock.Mock(side_effect=[('a' * 40).encode(), b''])
        ctx['identity']._read.return_value = b'changed'
        self.assertFalse(entry._git_check(ctx))

    def test_model_byte_change_rejects_stage(self):
        ctx, driver = fake_context()
        driver._read.return_value = ctx['manifest_raw']
        hashes = {ROOT / rel: digest for rel, digest in (*ctx['manifest'].panel.models, ctx['manifest'].panel.crop)}
        for role in ('market', 'round2'):
            hashes[STAGE / 'sources' / (role + '.py')] = ctx['manifest'].opponent(role)[1]
        driver._hash.side_effect = lambda _root, path, _limit: hashes[path]
        with mock.patch.object(Path, 'is_dir', autospec=True, return_value=True), \
             mock.patch.object(entry.os.path, 'lexists', return_value=False):
            self.assertTrue(entry._stage_check(ctx))
            hashes[ROOT / ctx['manifest'].panel.models[0][0]] = '0' * 64
            self.assertFalse(entry._stage_check(ctx))

    def test_input_preparation_uses_owned_budget_and_no_overwrite(self):
        ctx, driver = fake_context()
        ctx['archives'] = dict(tasks={role: {'data': ('toy-' + role).encode()}
                                      for role in ('baseline', 'control', 'auxiliary')},
                               crop={'data': b'toy-crop'})
        local = SimpleNamespace(stage=STAGE, claimed=True)
        written = {}
        driver._read.side_effect = lambda _root, path, _max: written[path]
        with mock.patch.object(entry.os.path, 'lexists', return_value=False), \
             mock.patch.object(Path, 'mkdir', autospec=True) as mkdir, \
             mock.patch.object(storage, 'owned_writes', return_value=nullcontext()) as budget, \
             mock.patch.object(entry, '_write_exact', side_effect=lambda path, data: written.update({path: data})), \
             mock.patch.object(entry, '_validate_model_choice') as choices:
            self.assertTrue(entry._prepare(ctx, ctx['manifest'], local, 1024))
        mkdir.assert_called_once_with(STAGE / 'models', exist_ok=False)
        self.assertEqual(set(budget.call_args.kwargs['allowed_paths']),
            {'models/baseline.npz', 'models/control.npz', 'models/auxiliary.npz',
             'models/crop.npz', entry._MANIFEST})
        self.assertEqual(len(written), 5)
        choices.assert_called_once_with(ctx)
        with mock.patch.object(entry.os.path, 'lexists', return_value=False), \
             mock.patch.object(Path, 'mkdir', autospec=True) as mkdir, \
             mock.patch.object(entry, '_write_exact') as write:
            with self.assertRaises(ValueError): entry._prepare(ctx, ctx['manifest'], local, 1)
            mkdir.assert_not_called()
            write.assert_not_called()

    def test_two_model_heads_receive_same_input_and_use_their_own_paths(self):
        ctx, _ = fake_context()
        ctx['task_schema'] = dict(schema_version='task', schema_sha256='synthetic',
                                  feature_names=['target.x'], shared_feature_names=['shared'])
        constructed, received = [], []
        def network(path, digest):
            constructed.append((path, digest))
            def select(inputs):
                received.append(inputs)
                return 'right' if path.name == 'baseline.npz' else 'left'
            return select
        with mock.patch.object(entry.importlib, 'import_module', return_value=SimpleNamespace(TaskNetwork=network)):
            entry._validate_model_choice(ctx)
        self.assertEqual(received[0], received[1])
        self.assertEqual([path.name for path, _ in constructed], ['baseline.npz', 'control.npz'])
        self.assertEqual([digest for _, digest in constructed],
                         [ctx['manifest'].panel.models[0][1], ctx['manifest'].panel.models[1][1]])

    def test_control_child_requires_accepted_baseline(self):
        ctx, driver = fake_context()
        started = dict(schema='task-pair-dev-campaign-start-v1', registration_commit=ctx['commit'],
                       max_games=24, attempt=1, retry_allowed=False)
        baseline = entry._ALLOWED[0]
        supervision = json.dumps(dict(schema='task-pair-dev-job-supervision-v1', game=baseline,
                                      retry_allowed=False, supervision={'passed': True})).encode()
        accepted = json.dumps(dict(schema='task-pair-dev-job-acceptance-v1', game=baseline,
                                   passed=True, retry_allowed=False,
                                   supervision_sha256=hashlib.sha256(supervision).hexdigest())).encode()
        files = {STAGE / 'campaign-started.local.json': json.dumps(started).encode(),
                 STAGE / 'job-results' / (baseline + '.supervision.local.json'): supervision}
        driver._read.side_effect = lambda _root, path, _max: files[path]
        with mock.patch.object(entry, '_stage_check', return_value=True), \
             mock.patch.object(entry, '_attempt_check', return_value=True), \
             mock.patch.object(entry.os.path, 'lexists', return_value=False):
            with self.assertRaises(KeyError): entry._child_ready(ctx, entry._ALLOWED[1])
            files[STAGE / 'job-results' / (baseline + '.acceptance.local.json')] = accepted
            entry._child_ready(ctx, entry._ALLOWED[1])

    def test_result_write_is_allowlisted_and_contains_no_event_rows_or_paths(self):
        ctx, driver = fake_context()
        written = {}
        for game in entry._ALLOWED:
            for kind in ('permit', 'consumed'):
                path, raw = entry._game_marker(ctx, game, kind)
                written[path] = raw
        driver._hash.return_value = 'b' * 64
        driver._comparison_file = lambda stage, game, arm: stage / (game + '-' + arm)
        driver._read.side_effect = lambda _root, path, _max: written[path]
        payload = dict(scope='synthetic_toy', attempted=2, official_replay_passes=2, comparisons=1,
                       promotion=False, games=[dict(game=game, secret='PRIVATE_EVENT') for game in entry._ALLOWED])
        def write(path, data): written[path] = data
        with mock.patch.object(storage, 'owned_writes', return_value=nullcontext()) as budget, \
             mock.patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=4 * 1024**2)), \
             mock.patch.object(entry, '_write_exact', side_effect=write):
            self.assertTrue(entry._save(ctx, payload))
        saved = json.loads(written[STAGE / entry._RESULT])
        rendered = json.dumps(saved)
        self.assertNotIn('PRIVATE_EVENT', rendered)
        self.assertNotIn(str(ROOT), rendered)
        self.assertEqual(saved['phase'], 'intermediate_requires_runner_return_and_observed_parent_exit')
        self.assertEqual(budget.call_args.kwargs['allowed_paths'], [entry._RESULT])

    def test_return_marker_never_claims_outer_exit(self):
        ctx, driver = fake_context()
        written = {}
        driver._read.side_effect = lambda _root, path, _max: (b'intermediate' if path.name == entry._RESULT
                                                               else written[path])
        result = dict(passed=True, scope='synthetic_toy', games=2,
                      official_replay_passes=2, comparisons=1, promotion=False)
        with mock.patch.object(entry, '_check', return_value=True), \
             mock.patch.object(storage, 'owned_writes', return_value=nullcontext()), \
             mock.patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=4 * 1024**2)), \
             mock.patch.object(entry, '_write_exact', side_effect=lambda path, data: written.update({path: data})):
            entry._mark_returned(ctx, result)
        marker = json.loads(written[STAGE / entry._RETURNED])
        self.assertEqual(marker['phase'], 'runner_returned')
        self.assertIs(marker['observed_outer_exit'], False)
        self.assertEqual(marker['intermediate_sha256'], hashlib.sha256(b'intermediate').hexdigest())

    def test_small_receipts_stop_before_write_if_stage_or_disk_limit_is_low(self):
        ctx, driver = fake_context()
        driver._owned_size.return_value = ctx['limits']['total_output_bytes']
        with mock.patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=4 * 1024**2)):
            with self.assertRaises(ValueError): entry._ensure_room(ctx, 1)
        driver._owned_size.return_value = 0
        with mock.patch.object(entry.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaises(ValueError): entry._ensure_room(ctx, 1)


if __name__ == '__main__':
    unittest.main()
