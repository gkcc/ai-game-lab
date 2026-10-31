"""Synthetic in-memory provenance and gzip tests; no real game files."""
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_pair_dev_campaign_v1.manifest import build_manifest
from learning.task_pair_dev_games_v1 import panel
from learning.task_pair_fresh_multi_v1 import entry, prepare as fresh
from . import samples, sources

REAL_OPEN_CHECKED = sources._open_checked


def encoded(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


class Fixture:
    def __init__(self):
        self.files, self.opened = {}, []
        self.binding_raw, self.market_raw = b'synthetic-binding', b'synthetic-market-binding'
        self.binding_sha = digest(self.binding_raw)
        self.roles = dict(zip(samples.ROLES, (101, 202, 303, 404)))
        self.binding = dict(seed_roles=self.roles, cohorts=deepcopy(entry.COHORTS),
                            market_binding_sha256=digest(self.market_raw))
        self.inventory = dict(market_binding_sha256=digest(self.market_raw))
        model_data = {name: ('synthetic-' + name).encode()
                      for name in ('baseline', 'control', 'auxiliary', 'crop')}
        self.market = dict(
            models={name: dict(path='artifacts/synthetic/' + name + '.npz', sha256=digest(model_data[name]))
                    for name in ('baseline', 'control', 'auxiliary')},
            crop=dict(path='artifacts/synthetic/crop.npz', sha256=digest(model_data['crop'])),
            opponents={name: dict(path='bots/' + name + '.py', sha256=char * 64)
                       for name, char in zip(('market', 'round2'), 'ef')}, worker_sha256='9' * 64)
        for name, data in model_data.items():
            self.files[str(fresh.GAME / 'artifacts' / 'synthetic' / (name + '.npz'))] = data
        self.registration = dict(schema=entry.SCHEMA, binding_commitment_sha256=self.binding_sha,
            engine_commit=panel.ENGINE_COMMIT, engine_sha256=panel.ENGINE_SHA,
            panel=deepcopy(entry.PANEL), evaluation=deepcopy(entry.EVALUATION))
        self.closed = dict(schema='task-pair-fresh-multi-closed-v1',
            registration_commit=sources.OLD_REGISTRATION_COMMIT, binding_sha256=self.binding_sha,
            outcome='completed', retry_allowed=False)
        self.campaign = dict(schema='task-pair-fresh-multi-private-result-v1',
            registration_commit=sources.OLD_REGISTRATION_COMMIT, binding_sha256=self.binding_sha,
            games=32, official_replay_passes=32, comparisons=16, prefix_comparisons=16,
            training=False, promotion=False, receipts=[dict(cohort=cohort, game=game,
                acceptance_sha256='0' * 64) for cohort in entry.COHORTS for game in entry.GAMES])
        for role in samples.ROLES:
            for opponent in samples.OPPONENTS:
                self.make_game(role, opponent)
        self.refresh_campaign()

    def refresh_campaign(self):
        self.files[str(fresh.PRIVATE / 'closed.local.json')] = encoded(self.closed)
        self.files[str(fresh.PRIVATE / 'campaign-result.local.json')] = encoded(self.campaign)
        self.campaign_sha = digest(self.files[str(fresh.PRIVATE / 'campaign-result.local.json')])

    def make_game(self, role, opponent):
        cohort, alias = next((c, a) for c, aliases in entry.COHORTS.items()
                             for a, r in aliases.items() if r == role)
        manifest = build_manifest(seeds={a: self.roles[r] for a, r in entry.COHORTS[cohort].items()},
                                  **self.market)
        coord = samples.source_coordinates(role, opponent)
        plan = next(p for p in manifest.panel.plans() if p['seed_alias'] == alias
                    and p['opponent'] == opponent and p['side'] == coord['side']
                    and p['policy'] == coord['arm'])
        name = plan['game']
        stage = fresh.ARTIFACTS / entry.STAGES[cohort]
        task_sha, crop_sha = manifest.panel.task_model(plan)[1], manifest.panel.crop[1]
        frames = []
        for step in range(720):
            status = 'DONE' if step == 719 else 'ACTIVE'
            frames.append([dict(observation=dict(step=step), action={'farmer': ['PASS'], 'hands': [], 'market': []},
                                status=status) for _ in range(2)])
        replay = dict(name='kaggriculture', info=dict(seed=plan['seed']),
                      configuration=dict(seed=None, episodeSteps=720), steps=frames)
        replay_raw = gzip.compress(encoded(replay), mtime=0)
        self.files[str(stage / 'games' / name / 'replay.json.gz')] = replay_raw
        trace = []
        task_forwards = task_rows = 0
        for step in range(719):
            turn = {key: 0 for key in samples.COUNT_KEYS}; turn['controller_calls'] = 1
            records = []
            if step in (17, 250, 500):
                records = [dict(chosen='student', teacher_label='teacher',
                                source='callback',
                                inputs=dict(candidates=[dict(id=k) for k in ('student', 'teacher', 'third')]))]
                turn['task_forwards'], turn['task_rows'] = 1, 3
                task_forwards += 1; task_rows += 3
            cumulative = dict(turn); cumulative['controller_calls'] = step + 1
            cumulative['task_forwards'], cumulative['task_rows'] = task_forwards, task_rows
            trace.append(encoded(dict(schema_version='task-choice-v1', step=step,
                task_checkpoint_sha256=task_sha, checkpoint_sha256=crop_sha,
                load_errors=0, inference_errors=0, fallbacks=0, records=records, crop_records=[],
                model_counts=cumulative, turn_model_counts=turn)) + b'\n')
        trace_raw = gzip.compress(b''.join(trace), mtime=0)
        self.files[str(stage / 'traces' / (name + '.jsonl.gz'))] = trace_raw
        row = dict(plan, bot_sha256='8' * 64, rival_sha256=manifest.opponent(opponent)[1],
                   task_checkpoint_sha256=task_sha, crop_checkpoint_sha256=crop_sha,
                   passed=True, states=720, action_turns=719, status=['DONE', 'DONE'], agent_errors=0,
                   engine_sha256=panel.ENGINE_SHA, banks=[100, 200], rewards=[100, 200],
                   replay_gzip_sha256=digest(replay_raw), trace_sha256=digest(trace_raw),
                   semantic_replay_sha256='6' * 64, semantic_replay_sha256_v2='7' * 64)
        official = dict(passed=True, official_transitions_reexecuted=719,
                        both_public_and_private_states_match_each_step=True,
                        inventory_order_checked_each_step=True, engine_sha256=panel.ENGINE_SHA,
                        semantic_sha256=row['semantic_replay_sha256'],
                        semantic_sha256_v2=row['semantic_replay_sha256_v2'], final_rewards=row['banks'])
        model_counts = dict(task_forwards=3, controller_calls=719)
        audit = dict(game=name, passed=True, errors=0, actor_actions_reproduced=719,
                     rival_actions_reproduced=719, trace_turns_reproduced=719,
                     trace_sha256=row['trace_sha256'], replay_sha256=row['replay_gzip_sha256'], official=official,
                     primary_model_counts=model_counts, live_model_counts=model_counts)
        row_raw, audit_raw = encoded(row), encoded(audit)
        complete_raw = encoded(dict(game=name, passed=True, row_sha256=digest(row_raw),
                                    certificate_sha256=digest(audit_raw)))
        hashes = dict(row=digest(row_raw), audit=digest(audit_raw), complete=digest(complete_raw))
        for kind, data in (('row', row_raw), ('audit', audit_raw), ('complete', complete_raw)):
            self.files[str(stage / 'job-results' / (name + '.' + kind + '.json'))] = data
        accepted = dict(schema='task-pair-dev-job-acceptance-v1', game=name, passed=True,
                        retry_allowed=False, receipt_sha256=hashes)
        accepted_raw = encoded(accepted)
        self.files[str(stage / 'job-results' / (name + '.acceptance.local.json'))] = accepted_raw
        receipt = next(r for r in self.campaign['receipts'] if (r['cohort'], r['game']) == (cohort, name))
        receipt['acceptance_sha256'] = digest(accepted_raw)

    @contextmanager
    def opened_source(self, _root, path, _maximum, _guard):
        self.opened.append(str(path))
        if str(path) not in self.files:
            raise sources.SourceError('synthetic source missing')
        with io.BytesIO(self.files[str(path)]) as stream:
            yield stream

    def patches(self):
        return (
            patch.object(sources, '_registration_blob', side_effect=lambda: encoded(self.registration)),
            patch.object(sources.fresh, 'verify_binding', side_effect=lambda expected:
                         (self.binding_raw, deepcopy(self.binding), deepcopy(self.inventory))
                         if expected == self.binding_sha else (_ for _ in ()).throw(ValueError('binding mismatch'))),
            patch.object(sources.fresh, 'load_verified_market_binding',
                         side_effect=lambda: (self.market_raw, deepcopy(self.market))),
            patch.object(sources, '_rule_sha', return_value=panel.ENGINE_SHA),
            patch.object(sources, '_open_checked', side_effect=self.opened_source),
        )


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        for item in self.fx.patches():
            item.start(); self.addCleanup(item.stop)

    def context(self):
        return sources.open_context(campaign_sha256=self.fx.campaign_sha, guard=lambda: None)

    def test_fixed_eight_coordinates_and_one_game_lifetime(self):
        ctx = self.context()
        self.assertEqual(repr(ctx), '<PrivateSourceContext>')
        for role in samples.ROLES:
            for opponent in samples.OPPONENTS:
                with sources.read_game(ctx, role=role, opponent=opponent, guard=lambda: None) as game:
                    self.assertEqual(repr(game), '<PrivateSourceGame>')
                    expected = samples.source_coordinates(role, opponent)
                    self.assertEqual({k: game.identity[k] for k in expected}, expected)
                    self.assertEqual([s['step'] for s in game.selected], [17, 250, 500])
                    self.assertEqual(len(game.replay['steps']), 720)
                    self.assertGreater(game.identity['trace_decoded_bytes'], 0)
                    self.assertTrue(Path(game.identity['task_path']).is_absolute())
                    self.assertTrue(Path(game.identity['crop_path']).is_absolute())
                    with self.assertRaises(sources.SourceError):
                        sources.read_game(ctx, role=role, opponent=opponent, guard=lambda: None)
                self.assertIsNone(game.replay)
                self.assertIsNone(game.selected)

    def test_commitment_campaign_and_closed_mismatch_stop(self):
        with self.assertRaises(sources.SourceError):
            sources.open_context(campaign_sha256='0' * 64, guard=lambda: None)
        self.fx.closed['outcome'] = 'failed'; self.fx.refresh_campaign()
        with self.assertRaises(sources.SourceError):
            self.context()
        self.fx.closed['outcome'] = 'completed'; self.fx.refresh_campaign()
        self.fx.binding['market_binding_sha256'] = '1' * 64
        with self.assertRaises(sources.SourceError):
            self.context()

    def test_campaign_duplicate_coordinate_is_rejected(self):
        self.fx.campaign['receipts'][0] = deepcopy(self.fx.campaign['receipts'][1])
        self.fx.refresh_campaign()
        with self.assertRaises(sources.SourceError):
            self.context()

    def test_acceptance_or_row_tamper_fails_before_replay_open(self):
        for suffix in ('.acceptance.local.json', '.row.json'):
            ctx = self.context()
            key = next(k for k in self.fx.files if k.endswith(suffix))
            original = self.fx.files[key]
            self.fx.files[key] = original + b' '
            matching = next(r for r in self.fx.campaign['receipts'] if r['game'] in key
                            and entry.STAGES[r['cohort']] in key)
            cohort, alias, opponent, side, arm = matching['cohort'], *matching['game'].split('-')[1:]
            role = entry.COHORTS[cohort][alias]
            self.fx.opened.clear()
            with self.assertRaises(sources.SourceError):
                sources.read_game(ctx, role=role, opponent=opponent, guard=lambda: None)
            self.assertFalse(any('replay.json.gz' in p for p in self.fx.opened))
            self.fx.files[key] = original

    def test_tampered_trace_and_bounded_replay_fail_closed(self):
        ctx = self.context()
        key = next(k for k in self.fx.files if k.endswith('.jsonl.gz'))
        original = self.fx.files[key]
        self.fx.files[key] = original + b'garbage'
        matching = next(r for r in self.fx.campaign['receipts'] if r['game'] in key
                        and entry.STAGES[r['cohort']] in key)
        parts = matching['game'].split('-')
        role, opponent = entry.COHORTS[matching['cohort']][parts[1]], parts[2]
        with self.assertRaises(sources.SourceError):
            sources.read_game(ctx, role=role, opponent=opponent, guard=lambda: None)
        self.assertFalse(ctx.active)
        self.fx.files[key] = original
        with patch.object(sources, 'REPLAY_DECODED_MAX', 1024), self.assertRaises(sources.SourceError):
            sources.read_game(ctx, role=role, opponent=opponent, guard=lambda: None)

    def test_model_bytes_are_checked_before_replay(self):
        ctx = self.context()
        key = next(k for k in self.fx.files if k.endswith('baseline.npz'))
        self.fx.files[key] += b'tampered'
        role, opponent = next((r, o) for r in samples.ROLES for o in samples.OPPONENTS
                              if samples.source_coordinates(r, o)['arm'] == 'baseline')
        self.fx.opened.clear()
        with self.assertRaises(sources.SourceError):
            sources.read_game(ctx, role=role, opponent=opponent, guard=lambda: None)
        self.assertFalse(any('replay.json.gz' in p for p in self.fx.opened))

    def test_game_identity_row_cannot_change_seed_or_model(self):
        ctx = self.context()
        cohort, alias, seed = ctx.role_map['F01']
        coord = samples.source_coordinates('F01', 'market')
        manifest = ctx.manifests[cohort]
        plan = next(p for p in manifest.panel.plans() if p['seed_alias'] == alias
                    and p['opponent'] == 'market' and p['side'] == coord['side'] and p['policy'] == coord['arm'])
        stage, name = fresh.ARTIFACTS / entry.STAGES[cohort], plan['game']
        row = json.loads(self.fx.files[str(stage / 'job-results' / (name + '.row.json'))])
        audit = json.loads(self.fx.files[str(stage / 'job-results' / (name + '.audit.json'))])
        complete = json.loads(self.fx.files[str(stage / 'job-results' / (name + '.complete.json'))])
        for key, wrong in (('seed', seed + 1), ('task_checkpoint_sha256', 'f' * 64),
                           ('engine_sha256', '0' * 64), ('action_turns', 718)):
            changed = dict(row, **{key: wrong})
            with self.subTest(key=key), self.assertRaises(ValueError):
                sources._validate_row_audit(changed, audit, complete, complete['row_sha256'],
                                            complete['certificate_sha256'], manifest, plan)

    def test_guard_failure_propagates(self):
        ctx = self.context()
        def stop():
            raise MemoryError('budget')
        with self.assertRaises(MemoryError):
            sources.read_game(ctx, role='F01', opponent='market', guard=stop)
        self.assertFalse(ctx.active)

    def test_open_checked_does_not_wrap_guard_oserror(self):
        class MemoryFile(io.BytesIO):
            def fileno(self):
                return 1
        fake = SimpleNamespace(st_mode=0o100600, st_dev=1, st_ino=1, st_size=3, st_mtime_ns=1)
        calls = []
        error = OSError('guard sentinel')
        def guard():
            calls.append(1)
            if len(calls) == 2:
                raise error
        path = Path('synthetic')
        with patch.object(sources.fresh, '_inside', return_value=path), \
             patch.object(Path, 'lstat', return_value=fake), \
             patch.object(Path, 'open', return_value=MemoryFile(b'abc')), \
             patch.object(sources.os, 'fstat', return_value=fake):
            with self.assertRaises(OSError) as raised:
                with REAL_OPEN_CHECKED(path, path, 8, guard):
                    pass
        self.assertIs(raised.exception, error)

    def test_underlying_read_oserror_does_not_expose_path(self):
        class FailedRead(io.BytesIO):
            def read(self, _size=-1):
                raise OSError('private-path-sentinel')
        with self.assertRaises(sources.SourceError) as raised:
            sources._hash_stream(FailedRead(b'data'), 100, lambda: None)
        self.assertNotIn('private-path-sentinel', str(raised.exception))

    def test_trace_changed_on_same_descriptor_after_scan_is_rejected(self):
        ctx = self.context()
        target = next(k for k in self.fx.files if k.endswith('.jsonl.gz'))
        matching = next(r for r in self.fx.campaign['receipts'] if r['game'] in target
                        and entry.STAGES[r['cohort']] in target)
        parts = matching['game'].split('-')
        role, opponent = entry.COHORTS[matching['cohort']][parts[1]], parts[2]

        class ChangingFile(io.BytesIO):
            def __init__(self, raw):
                super().__init__(raw)
                self.zero_seeks = 0

            def seek(self, offset, whence=0):
                if offset == 0 and whence == 0:
                    self.zero_seeks += 1
                    if self.zero_seeks == 3:
                        super().seek(0, 2)
                        super().write(b'changed')
                return super().seek(offset, whence)

        @contextmanager
        def source(_root, path, _maximum, _guard):
            raw = self.fx.files[str(path)]
            with (ChangingFile(raw) if str(path) == target else io.BytesIO(raw)) as stream:
                yield stream

        with patch.object(sources, '_open_checked', side_effect=source), self.assertRaises(sources.SourceError):
            sources.read_game(ctx, role=role, opponent=opponent, guard=lambda: None)


if __name__ == '__main__':
    unittest.main()
