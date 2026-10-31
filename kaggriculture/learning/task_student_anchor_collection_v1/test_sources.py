"""In-memory Anchor provenance and lifecycle regressions; no real games."""
from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from learning.task_student_dagger_collection_v1 import sources as dagger
from learning.task_student_dagger_collection_v1.test_sources import Fixture as DaggerFixture, digest, encoded
from . import sources


class Fixture:
    def __init__(self):
        self.dagger = DaggerFixture()
        self.files, self.blobs = self.dagger.files, {}
        self.stage = sources.old.fresh.ARTIFACTS / sources.STAGE_NAME
        self.dagger_stage = sources.old.fresh.ARTIFACTS / dagger.STAGE_NAME
        self.task_sha, self.crop_sha = self.dagger.task_sha, self.dagger.crop_sha
        self.anchor_rows, self.job_hashes = [], {}
        for opponent in sources.OPPONENTS:
            self.files[str(self.stage/'sources'/(opponent+'.py'))] = ('synthetic-opponent-'+opponent).encode()
        for split, seeds in (('dev', (200, 201)), ('holdout', tuple(range(210, 218)))):
            for seed in seeds:
                for opponent in sources.OPPONENTS:
                    for side in (0, 1):
                        self.add_game(split, seed, opponent, side)

        self.fresh_roles = dict(zip(('F01', 'F02', 'F03', 'F04'), range(300, 304)))
        binding_path = sources.old.fresh.PRIVATE/'binding.local.json'
        binding = json.loads(self.files[str(binding_path)])
        binding.update(seed_roles=self.fresh_roles, cohorts=sources.old.entry.COHORTS)
        binding_raw = encoded(binding)
        self.files[str(binding_path)] = binding_raw
        reg_key = (dagger.FRESH_COMMIT, dagger.FRESH_REGISTRATION)
        reg = json.loads(self.dagger.blobs[reg_key])
        reg.update(schema=sources.old.entry.SCHEMA, panel=sources.old.entry.PANEL,
                   evaluation=sources.old.entry.EVALUATION,
                   binding_commitment_sha256=digest(binding_raw))
        self.dagger.blobs[reg_key] = encoded(reg)
        receipts = [dict(cohort=cohort, game=game, acceptance_sha256='a'*64)
                    for cohort in sources.old.entry.COHORTS for game in sources.old.entry.GAMES]
        campaign = dict(schema='task-pair-fresh-multi-private-result-v1',
            registration_commit=dagger.FRESH_COMMIT, binding_sha256=digest(binding_raw),
            games=32, official_replay_passes=32, comparisons=16, prefix_comparisons=16,
            training=False, promotion=False, receipts=receipts)
        closed = dict(schema='task-pair-fresh-multi-closed-v1', registration_commit=dagger.FRESH_COMMIT,
                      binding_sha256=digest(binding_raw), outcome='completed', retry_allowed=False)
        self.files[str(sources.old.fresh.PRIVATE/'campaign-result.local.json')] = encoded(campaign)
        self.files[str(sources.old.fresh.PRIVATE/'closed.local.json')] = encoded(closed)
        self.blobs[(sources.FRESH_RESULT_COMMIT, sources.FRESH_RESULT)] = encoded(dict(
            schema='task-pair-fresh-multi-public-v1', registration_commit=dagger.FRESH_COMMIT,
            games=32, official_replays=32, seed_clusters=4, training=False, promotion=False,
            by_seed={role: {} for role in self.fresh_roles}))
        self.rebuild()

    def add_game(self, split, seed, opponent, side):
        info = self.dagger.make_game(split, seed, opponent, side)
        game = info['game']
        paths = {
            'input': ('job-inputs', game+'.json'),
            'row': ('job-results', game+'.row.json'),
            'audit': ('job-results', game+'.audit.json'),
            'complete': ('job-results', game+'.complete.json'),
            'result': ('games', game, 'result.json'),
            'replay': ('games', game, 'replay.json.gz'),
            'trace': ('traces', game+'.jsonl.gz'),
            'wrapper': ('agents', game+'.py'),
        }
        for parts in paths.values():
            self.files[str(self.stage.joinpath(*parts))] = self.files[str(self.dagger_stage.joinpath(*parts))]
        job_path = str(self.stage.joinpath(*paths['input']))
        row_path = str(self.stage.joinpath(*paths['row']))
        complete_path = str(self.stage.joinpath(*paths['complete']))
        rival = digest(self.files[str(self.stage/'sources'/(opponent+'.py'))])
        job, row = json.loads(self.files[job_path]), json.loads(self.files[row_path])
        job['rival_sha256'] = row['rival_sha256'] = rival
        self.files[job_path], self.files[row_path] = encoded(job), encoded(row)
        complete = json.loads(self.files[complete_path]); complete['row_sha256'] = digest(self.files[row_path])
        self.files[complete_path] = encoded(complete)
        storage_path = str(self.stage/'job-results'/(game+'.storage.json'))
        self.files[storage_path] = encoded(dict(passed=True, error_kind=None,
                                               output=dict(refused=0, used_bytes=100, limit_bytes=1000)))
        self.job_hashes[game] = {kind: digest(self.files[str(self.stage/'job-results'/(game+'.'+kind+'.json'))])
                                 for kind in ('row', 'audit', 'complete', 'storage')}
        self.anchor_rows.append(row)

    def rebuild(self):
        dummy = {'dummy'+str(i): {kind:'b'*64 for kind in sources.RECEIPTS} for i in range(80)}
        stats = encoded(dict(passed=True, all_job_evidence_sha256={**self.job_hashes, **dummy}))
        freeze = self.dagger.blobs[(dagger.RESULT_COMMIT, dagger.RESULT_PREFIX+'freeze.json')]
        report = encoded(dict(review_passed=True, all_registered_jobs_completed=True, completed_games=120,
                              final_statistics_sha256=digest(stats), freeze_sha256=digest(freeze)))
        verification = encoded(dict(passed=True, report_sha256=digest(report), statistics_sha256=digest(stats)))
        root = encoded(dict(passed=True, engineering_complete=True, completed_games=120,
            independent_verification_sha256=digest(verification), independent_result_sha256=digest(report),
            freeze_sha256=digest(freeze)))
        self.root_sha = digest(root)
        self.files[str(self.stage/'independent/final-result-review.json')] = report
        self.files[str(self.stage/'independent/final-statistics.json')] = stats
        for name, raw in (('root-result-acceptance.json', root),
                          ('independent/final-verification.json', verification), ('freeze.json', freeze),
                          ('ledger.json', encoded(self.anchor_rows + [{'policy':'candidate'}]*80))):
            self.blobs[(sources.ANCHOR_COMMIT, sources.ANCHOR_PREFIX+name)] = raw

    @contextmanager
    def open_file(self, _root, path, _maximum, _guard):
        if str(path) not in self.files:
            raise sources.SourceError('synthetic source missing')
        with io.BytesIO(self.files[str(path)]) as stream:
            yield stream


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        patches = (
            patch.object(sources, 'ANCHOR_ROOT_SHA256', self.fx.root_sha),
            patch.object(sources, '_git_blob', side_effect=lambda commit, relative, _guard:
                         self.fx.blobs[(commit, relative)]),
            patch.object(dagger, 'ROOT_SHA256', self.fx.dagger.root_sha),
            patch.object(dagger, '_git_blob', side_effect=lambda commit, relative, _guard:
                         self.fx.dagger.blobs[(commit, relative)]),
            patch.object(sources.old, '_open_checked', side_effect=self.fx.open_file),
            patch.object(sources.old, '_rule_sha', return_value=dagger.ENGINE_SHA),
            patch.object(sources.old.fresh, 'verify_binding', side_effect=AssertionError('reserved read')),
            patch.object(sources.old.fresh, 'load_verified_market_binding', side_effect=AssertionError('reserved read')),
        )
        for item in patches:
            item.start(); self.addCleanup(item.stop)

    def context(self):
        return sources.open_context(guard=lambda: None)

    def game(self, context, *, batch=1, role='T01', opponent='market', side=0):
        return sources.read_game(context, batch=batch, role=role, opponent=opponent, side=side, guard=lambda: None)

    def test_ten_stable_roles_preserve_historical_development_and_empty_slots(self):
        context = self.context()
        roles = {1: dict(zip(sources.TRAIN_ROLES + sources.DEV_ROLES,
                             ('A01','A02','A03','A04','A07','A08'))),
                 2: dict(zip(sources.TRAIN_ROLES[:2] + sources.DEV_ROLES,
                             ('A05','A06','A09','A10')))}
        for batch, slots in roles.items():
            for role, source_role in slots.items():
                self.assertEqual(context.source_role(batch=batch, role=role), source_role)
                with self.game(context, batch=batch, role=role) as game:
                    self.assertEqual(game.identity['source_role'], source_role)
                    self.assertEqual(game.identity['split'], 'train' if role.startswith('T') else 'dev')
                    self.assertEqual(game._item.split, 'dev' if source_role in ('A07','A08') else 'holdout')
        binding = context.private_role_binding()
        self.assertEqual(set(binding), set(sources.SOURCE_ROLES))
        self.assertEqual(len(set(binding.values())), 10)
        self.assertEqual(context.task_sha256, self.fx.task_sha)
        self.assertEqual(context.crop_sha256, self.fx.crop_sha)
        binding['A01'] = None
        self.assertIsNotNone(context.private_role_binding()['A01'])
        for role in ('T03', 'T04'):
            self.assertIsNone(context.source_role(batch=2, role=role))
            with self.assertRaises(sources.SourceError):
                self.game(context, batch=2, role=role)
        with self.assertRaises(sources.SourceError):
            self.game(context, batch=3)

    def test_root_and_private_statistics_mutations_fail_closed(self):
        key = (sources.ANCHOR_COMMIT, sources.ANCHOR_PREFIX+'independent/final-verification.json')
        original = self.fx.blobs[key]; self.fx.blobs[key] += b' '
        with self.assertRaises(sources.SourceError): self.context()
        self.fx.blobs[key] = original
        path = str(self.fx.stage/'independent/final-statistics.json')
        self.fx.files[path] += b' '
        with self.assertRaises(sources.SourceError): self.context()

    def test_incomplete_or_duplicate_completed_ledger_fails(self):
        key = (sources.ANCHOR_COMMIT, sources.ANCHOR_PREFIX+'ledger.json')
        doc = json.loads(self.fx.blobs[key])
        doc[0] = deepcopy(doc[1])
        self.fx.blobs[key] = encoded(doc)
        with self.assertRaises(sources.SourceError): self.context()

    def test_overlap_with_completed_dagger_or_fresh_fails(self):
        anchor_seed = self.fx.anchor_rows[0]['seed']
        dagger_seed = self.fx.dagger.games[0]['seed']
        with patch.object(sources, '_fresh_completed_seeds', return_value={dagger_seed}):
            with self.assertRaisesRegex(sources.SourceError, 'completed historical roles overlap'): self.context()
        with patch.object(sources, '_fresh_completed_seeds', return_value={anchor_seed}):
            with self.assertRaisesRegex(sources.SourceError, 'historical completed roles overlap'): self.context()
        with patch.object(sources, '_fresh_completed_seeds', return_value=set()):
            with patch.object(dagger, 'open_context') as opened:
                entries = {(i, 'x', 0): dagger._Entry('x', seed, 'dev', 'market', 0, {})
                           for i, seed in enumerate((anchor_seed, 400, 401, 402, 403, 404))}
                base = dagger.PrivateContext(entries, 'unused', self.fx.task_sha, 'unused', self.fx.crop_sha)
                opened.return_value = base
                with self.assertRaisesRegex(sources.SourceError, 'historical completed roles overlap'): self.context()

    def test_all_four_receipts_replay_trace_and_wrapper_are_hash_bound(self):
        row = next(r for r in self.fx.anchor_rows if r['split']=='holdout'
                   and r['opponent']=='market' and r['side']==0)
        game = row['game']; stage = self.fx.stage
        paths = [stage/'job-results'/(game+'.'+kind+'.json') for kind in ('row','audit','complete','storage')]
        paths += [stage/'games'/game/'replay.json.gz', stage/'traces'/(game+'.jsonl.gz'),
                  stage/'agents'/(game+'.py'), stage/'sources'/'market.py']
        for path in paths:
            ctx = self.context(); key = str(path); old = self.fx.files[key]
            self.fx.files[key] = old + b' '
            with self.assertRaises(sources.SourceError): self.game(ctx)
            self.fx.files[key] = old
            with self.assertRaises(sources.SourceError): self.game(ctx)

    def test_unsigned_input_and_result_require_matching_semantics(self):
        row = next(r for r in self.fx.anchor_rows if r['split']=='holdout'
                   and r['opponent']=='market' and r['side']==0)
        game = row['game']
        for path, field, value in ((self.fx.stage/'job-inputs'/(game+'.json'), 'side', 1),
                                   (self.fx.stage/'games'/game/'result.json', 'done', False)):
            ctx = self.context(); key = str(path); old = self.fx.files[key]
            document = json.loads(old); document[field] = value; self.fx.files[key] = encoded(document)
            with self.assertRaises(sources.SourceError): self.game(ctx)
            self.fx.files[key] = old

    def test_semantically_changed_public_ledger_row_is_not_accepted(self):
        key = (sources.ANCHOR_COMMIT, sources.ANCHOR_PREFIX+'ledger.json')
        doc = json.loads(self.fx.blobs[key])
        row = next(r for r in doc if r.get('policy')=='baseline' and r['split']=='holdout'
                   and r['opponent']=='market' and r['side']==0)
        row['net_profit'] = 1
        self.fx.blobs[key] = encoded(doc)
        with self.assertRaises(sources.SourceError): self.game(self.context())

    def test_two_pass_limit_requires_normal_semantic_completion(self):
        ctx = self.context()
        for _ in range(2):
            with self.game(ctx) as game:
                accounting, windows = game.scan_candidates(guard=lambda: None)
                self.assertEqual(accounting['lines'], 719)
                self.assertEqual(sum(map(len, windows)), 3)
        with self.assertRaises(sources.SourceError): self.game(ctx)
        with self.game(ctx, batch=2, role='T01') as game:
            self.assertEqual(game.identity['source_role'], 'A05')

        ctx = self.context()
        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            with self.game(ctx) as game:
                game.scan_candidates(guard=lambda: None)
                raise RuntimeError('synthetic failure')
        with self.assertRaises(sources.SourceError): self.game(ctx)

        ctx = self.context()
        with self.game(ctx) as game:
            with game.open_trace(guard=lambda: None) as trace:
                for _ in range(719): self.assertTrue(trace.readline(dagger.MAX_LINE+1))
                self.assertEqual(trace.read(1), b'')
        with self.assertRaises(sources.SourceError): self.game(ctx)


if __name__ == '__main__':
    unittest.main()
