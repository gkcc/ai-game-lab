"""In-memory source authentication tests; no real traces, games or models."""
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

import numpy as np

from learning.task_student_continuation_v1 import samples
from . import sources


def encoded(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class Fixture:
    def __init__(self):
        self.files, self.blobs = {}, {}
        self.task_raw, self.crop_raw = b'synthetic-task', b'synthetic-crop'
        self.task_sha, self.crop_sha = digest(self.task_raw), digest(self.crop_raw)
        self.files[str(sources.old.fresh.GAME / 'artifacts/synthetic/task.npz')] = self.task_raw
        self.files[str(sources.old.fresh.GAME / 'artifacts/synthetic/crop.npz')] = self.crop_raw
        market = dict(schema='task-pair-market-consumed-binding-v1',
            models=dict(baseline=dict(path='artifacts/synthetic/task.npz', sha256=self.task_sha)),
            crop=dict(path='artifacts/synthetic/crop.npz', sha256=self.crop_sha))
        market_raw = encoded(market)
        market_path = sources.old.fresh.PRIVATE_ROOT / 'task-pair-market-consumed-v1-20261031' / 'binding.local.json'
        self.files[str(market_path)] = market_raw
        fresh_raw = encoded(dict(schema='task-pair-fresh-multi-private-binding-v1',
                                 market_binding_sha256=digest(market_raw)))
        self.files[str(sources.old.fresh.PRIVATE / 'binding.local.json')] = fresh_raw
        self.blobs[(sources.FRESH_COMMIT, sources.FRESH_REGISTRATION)] = encoded(dict(
            engine_commit=sources.ENGINE_COMMIT, engine_sha256=sources.ENGINE_SHA,
            binding_commitment_sha256=digest(fresh_raw)))

        runtime, public = {}, {}
        for i, name in enumerate(sources.RUNTIME_COMPONENTS):
            raw = ('synthetic-source-' + str(i)).encode()
            local = sources.old.fresh.GAME / name
            published = sources.old.fresh.REPO / 'kaggriculture' / name
            self.files[str(local)] = raw; self.files[str(published)] = raw
            runtime[name] = digest(raw); public[name] = digest(raw)
        freeze_raw = encoded(dict(runtime_files=runtime, published_runtime_files=public))
        self.blobs[(sources.RESULT_COMMIT, sources.RESULT_PREFIX + 'freeze.json')] = freeze_raw

        self.games = []
        for split, role_count in (('train', 4), ('dev', 2)):
            for i in range(role_count):
                seed = 100 + i + (0 if split == 'train' else 10)
                for opponent in sources.OPPONENTS:
                    for side in (0, 1):
                        self.games.append(self.make_game(split, seed, opponent, side))
        collection_raw = encoded(dict(passed=True, exact_registered_population_complete=True,
                                      reviewed_games=deepcopy(self.games)))
        review_raw = encoded(dict(passed=True, certificate_sha256={
            'independent-collection-review.json': digest(collection_raw)}))
        root_raw = encoded(dict(passed=True, official_games=24, candidate_admission=False,
            freeze_sha256=digest(freeze_raw), independent_result_review_sha256=digest(review_raw)))
        self.root_sha = digest(root_raw)
        for name, raw in (('root-result-acceptance.json', root_raw),
                          ('independent-result-review.json', review_raw),
                          ('independent-collection-review.json', collection_raw)):
            self.blobs[(sources.RESULT_COMMIT, sources.RESULT_PREFIX + name)] = raw

    def make_game(self, split, seed, opponent, side):
        # Synthetic game keys are never printed or returned by the reader.
        game = f'synthetic_{split}_{seed}_{opponent}_{side}'
        stage = sources.old.fresh.ARTIFACTS / sources.STAGE_NAME
        wrapper = ('from learning.task_dagger_once_v1.policy import BoundedAgent\n'
                   '_runner = BoundedAgent(None,' + repr(self.crop_sha) + ',None,' + repr(self.task_sha)
                   + ',limits=' + repr(sources.GAME_LIMITS) + ')\n').encode()
        wrapper_sha = digest(wrapper)
        job = dict(game=game, seed=seed, split=split, policy='baseline', opponent=opponent,
                   side=side, parity_kind=None, task_checkpoint_sha256=self.task_sha,
                   bot_sha256=wrapper_sha, rival_sha256='e' * 64)
        trace = []
        task_calls = task_rows = 0
        for step in range(719):
            turn = {k: 0 for k in samples.COUNT_KEYS}; turn['controller_calls'] = 1
            records = []
            if step in (17, 250, 500):
                turn['task_forwards'], turn['task_rows'] = 1, 3
                task_calls += 1; task_rows += 3
                records = [dict(source='callback', chosen='student', teacher_label='teacher',
                                inputs=dict(candidates=[dict(id=k) for k in ('student', 'teacher', 'third')]))]
            total = dict(turn); total['controller_calls'] = step + 1
            total['task_forwards'], total['task_rows'] = task_calls, task_rows
            trace.append(encoded(dict(schema_version='task-choice-v1', step=step,
                task_checkpoint_sha256=self.task_sha, checkpoint_sha256=self.crop_sha,
                load_errors=0, inference_errors=0, fallbacks=0, records=records, crop_records=[],
                model_counts=total, turn_model_counts=turn)) + b'\n')
        trace_raw = gzip.compress(b''.join(trace), mtime=0)
        frames = [[dict(observation=dict(step=step), action=dict(farmer=['PASS'], hands=[], market=[]),
                        status='DONE' if step == 719 else 'ACTIVE') for _ in range(2)]
                  for step in range(720)]
        replay_raw = gzip.compress(encoded(dict(name='kaggriculture', info=dict(seed=seed),
            configuration=dict(seed=None, episodeSteps=720), steps=frames)), mtime=0)
        row = dict(job, crop_checkpoint_sha256=self.crop_sha, engine_sha256=sources.ENGINE_SHA,
            passed=True, states=720, action_turns=719, status=['DONE', 'DONE'], agent_errors=0,
            replay_gzip_sha256=digest(replay_raw), trace_sha256=digest(trace_raw),
            semantic_replay_sha256='6' * 64, semantic_replay_sha256_v2='7' * 64,
            banks=[100, 200], rewards=[100, 200])
        official = dict(passed=True, engine_sha256=sources.ENGINE_SHA, official_transitions_reexecuted=719,
            both_public_and_private_states_match_each_step=True, inventory_order_checked_each_step=True,
            semantic_sha256=row['semantic_replay_sha256'], semantic_sha256_v2=row['semantic_replay_sha256_v2'],
            final_rewards=row['banks'])
        counts = dict(task_forwards=3, controller_calls=719)
        audit = dict(game=game, passed=True, errors=0, actor_actions_reproduced=719,
            rival_actions_reproduced=719, trace_turns_reproduced=719, trace_sha256=row['trace_sha256'],
            replay_sha256=row['replay_gzip_sha256'], official=official,
            primary_model_counts=counts, live_model_counts=counts)
        row_raw, audit_raw = encoded(row), encoded(audit)
        complete_raw = encoded(dict(game=game, passed=True, row_sha256=digest(row_raw),
                                    certificate_sha256=digest(audit_raw)))
        result_raw = encoded(dict(passed=True, done=True, statuses=['DONE', 'DONE'], recorded_states=720,
            action_turns=719, agent_errors=[], engine_matches_official_commit=True,
            engine_sha256=sources.ENGINE_SHA, upstream_commit=sources.ENGINE_COMMIT,
            seed=seed, bot_side=side, final_money=row['banks'], rewards=row['rewards']))
        data = dict(input=encoded(job), row=row_raw, audit=audit_raw, completion=complete_raw,
                    result=result_raw, replay=replay_raw, trace=trace_raw, wrapper=wrapper)
        paths = dict(input=stage/'job-inputs'/(game+'.json'), row=stage/'job-results'/(game+'.row.json'),
            audit=stage/'job-results'/(game+'.audit.json'), completion=stage/'job-results'/(game+'.complete.json'),
            result=stage/'games'/game/'result.json', replay=stage/'games'/game/'replay.json.gz',
            trace=stage/'traces'/(game+'.jsonl.gz'), wrapper=stage/'agents'/(game+'.py'))
        for key in data:
            self.files[str(paths[key])] = data[key]
        return dict(game=game, seed=seed, split=split, opponent=opponent, side=side,
                    passed=True, policy='baseline', input_sha256={k:digest(v) for k,v in data.items()})

    def rebuild_root(self):
        collection_raw = encoded(dict(passed=True, exact_registered_population_complete=True,
                                      reviewed_games=deepcopy(self.games)))
        review_raw = encoded(dict(passed=True, certificate_sha256={
            'independent-collection-review.json': digest(collection_raw)}))
        root_raw = encoded(dict(passed=True, official_games=24, candidate_admission=False,
            freeze_sha256=digest(self.blobs[(sources.RESULT_COMMIT, sources.RESULT_PREFIX + 'freeze.json')]),
            independent_result_review_sha256=digest(review_raw)))
        self.root_sha = digest(root_raw)
        for name, raw in (('root-result-acceptance.json', root_raw),
                          ('independent-result-review.json', review_raw),
                          ('independent-collection-review.json', collection_raw)):
            self.blobs[(sources.RESULT_COMMIT, sources.RESULT_PREFIX + name)] = raw

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
            patch.object(sources, 'ROOT_SHA256', self.fx.root_sha),
            patch.object(sources, '_git_blob', side_effect=lambda commit, relative, _guard:
                         self.fx.blobs[(commit, relative)]),
            patch.object(sources.old, '_open_checked', side_effect=self.fx.open_file),
            patch.object(sources.old, '_rule_sha', return_value=sources.ENGINE_SHA),
            patch.object(sources.old.fresh, 'verify_binding', side_effect=AssertionError('reserved read')),
            patch.object(sources.old.fresh, 'load_verified_market_binding', side_effect=AssertionError('reserved read')),
        )
        for item in patches:
            item.start(); self.addCleanup(item.stop)

    def context(self):
        return sources.open_context(guard=lambda: None)

    def test_complete_role_panel_safe_repr_and_one_game_at_a_time(self):
        context = self.context()
        self.assertEqual(repr(context), '<PrivateDaggerSourceContext>')
        for role in sources.ROLES:
            for opponent in sources.OPPONENTS:
                for side in (0, 1):
                    with sources.read_game(context, role=role, opponent=opponent, side=side,
                                           guard=lambda: None) as game:
                        self.assertEqual(repr(game), '<PrivateDaggerSourceGame>')
                        self.assertEqual(game.identity, dict(role=role, opponent=opponent, side=side,
                            split='train' if role in sources.TRAIN_ROLES else 'dev'))
                        self.assertEqual(len(game.replay['steps']), 720)
                        self.assertEqual(game.game_limits, sources.GAME_LIMITS)
                        with self.assertRaises(sources.SourceError):
                            sources.read_game(context, role=role, opponent=opponent, side=side, guard=lambda: None)
                    with self.assertRaises(sources.SourceError):
                        _ = game.replay

    def test_scanner_must_finish_and_trace_is_single_use(self):
        context = self.context()
        with sources.read_game(context, role='T01', opponent='market', side=0, guard=lambda: None) as game:
            with game.open_trace(guard=lambda: None) as trace:
                found = samples.scan(trace, task_sha256=game.task_sha256,
                                     crop_sha256=game.crop_sha256, guard=lambda: None)
                self.assertEqual(found['lines'], 719)
                self.assertEqual([x['step'] for x in found['selected']], [17, 250, 500])
                with self.assertRaises(sources.SourceError):
                    game.close()
            with self.assertRaises(sources.SourceError):
                with game.open_trace(guard=lambda: None):
                    pass
        with self.assertRaises(sources.SourceError):
            sources.read_game(context, role='T01', opponent='market', side=0, guard=lambda: None)

    def test_two_complete_visits_then_no_third(self):
        context = self.context()
        for _ in range(2):
            with sources.read_game(context, role='T01', opponent='market', side=0, guard=lambda: None) as game:
                accounting, windows = game.scan_candidates(guard=lambda: None)
                self.assertEqual(accounting['lines'], 719)
                self.assertGreater(accounting['decoded_bytes'], 0)
                self.assertEqual([entry['step'] for window in windows for entry in window], [17, 250, 500])
        with self.assertRaises(sources.SourceError):
            sources.read_game(context, role='T01', opponent='market', side=0, guard=lambda: None)

    def test_manual_full_scan_cannot_authorize_a_second_visit(self):
        context = self.context()
        with sources.read_game(context, role='T01', opponent='market', side=0, guard=lambda: None) as game:
            with game.open_trace(guard=lambda: None) as trace:
                for _ in range(719):
                    self.assertTrue(trace.readline(sources.MAX_LINE + 1))
                self.assertEqual(trace.read(1), b'')
        with self.assertRaises(sources.SourceError):
            sources.read_game(context, role='T01', opponent='market', side=0, guard=lambda: None)

    def test_exception_after_valid_scan_cannot_authorize_a_second_visit(self):
        context = self.context()
        with self.assertRaisesRegex(RuntimeError, 'synthetic body failure'):
            with sources.read_game(context, role='T01', opponent='market', side=0, guard=lambda: None) as game:
                game.scan_candidates(guard=lambda: None)
                raise RuntimeError('synthetic body failure')
        with self.assertRaises(sources.SourceError):
            sources.read_game(context, role='T01', opponent='market', side=0, guard=lambda: None)

    def test_incomplete_or_failed_first_visit_cannot_be_retried(self):
        context = self.context()
        with sources.read_game(context, role='T01', opponent='market', side=0, guard=lambda: None):
            pass
        with self.assertRaises(sources.SourceError):
            sources.read_game(context, role='T01', opponent='market', side=0, guard=lambda: None)

        other = self.context()
        def fail_guard():
            raise RuntimeError('synthetic guard failure')
        with self.assertRaises(RuntimeError):
            sources.read_game(other, role='T01', opponent='market', side=0, guard=fail_guard)
        with self.assertRaises(sources.SourceError):
            sources.read_game(other, role='T01', opponent='market', side=0, guard=lambda: None)

    def test_commitment_and_certificate_mutation_fails(self):
        key = (sources.RESULT_COMMIT, sources.RESULT_PREFIX+'independent-result-review.json')
        original = self.fx.blobs[key]
        self.fx.blobs[key] += b' '
        with self.assertRaises(sources.SourceError):
            self.context()
        self.fx.blobs[key] = original
        self.fx.files[str(sources.old.fresh.PRIVATE/'binding.local.json')] += b' '
        with self.assertRaises(sources.SourceError):
            self.context()

    def test_duplicate_or_missing_completed_coordinate_fails(self):
        self.fx.games[0] = deepcopy(self.fx.games[1]); self.fx.rebuild_root()
        with patch.object(sources, 'ROOT_SHA256', self.fx.root_sha), self.assertRaises(sources.SourceError):
            self.context()

    def test_tampering_any_of_eight_files_stops_before_success(self):
        first = self.fx.games[0]; name = first['game']
        stage = sources.old.fresh.ARTIFACTS / sources.STAGE_NAME
        files = [stage/'job-inputs'/(name+'.json'), stage/'job-results'/(name+'.row.json'),
            stage/'job-results'/(name+'.audit.json'), stage/'job-results'/(name+'.complete.json'),
            stage/'games'/name/'result.json', stage/'games'/name/'replay.json.gz',
            stage/'traces'/(name+'.jsonl.gz'), stage/'agents'/(name+'.py')]
        for path in files:
            ctx = self.context()
            original = self.fx.files[str(path)]
            self.fx.files[str(path)] = original + b' '
            with self.subTest(kind=path.suffix), self.assertRaises(sources.SourceError):
                sources.read_game(ctx, role='T01', opponent='market', side=0, guard=lambda: None)
            self.fx.files[str(path)] = original
            self.assertFalse(ctx._active)

    def test_wrong_limits_or_model_file_fails(self):
        ctx = self.context()
        wrong = ('from learning.task_dagger_once_v1.policy import BoundedAgent\n'
                 '_runner=BoundedAgent(None,'+repr(self.fx.crop_sha)+',None,'+repr(self.fx.task_sha)
                 +',limits={})\n').encode()
        with self.assertRaises(sources.SourceError):
            sources._verify_wrapper(wrong, self.fx.task_sha, self.fx.crop_sha)
        path = str(sources.old.fresh.GAME/'artifacts/synthetic/task.npz')
        self.fx.files[path] += b'changed'
        with self.assertRaises(ValueError):
            self.context()

    def test_normalization_returns_copies_and_model_factory_is_mocked(self):
        ctx = self.context()
        network = SimpleNamespace(mean=np.array([1.]), scale=np.array([2.]), metadata={'nested': [3]})
        with patch('learning.task_bc_v1.model.TaskNetwork', return_value=network) as ctor:
            mean, scale, metadata = ctx.model_normalization()
            mean[0] = 99; scale[0] = 99; metadata['nested'][0] = 99
            self.assertEqual(network.mean[0], 1.); self.assertEqual(network.scale[0], 2.)
            self.assertEqual(network.metadata, {'nested': [3]})
            self.assertEqual(ctx._visits, {})
            ctor.assert_called_once_with(ctx._task_path, ctx._task_sha)
        with sources.read_game(ctx, role='T01', opponent='market', side=0, guard=lambda: None) as game:
            with patch('learning.task_bc_v1.model.TaskNetwork', return_value=network) as ctor:
                mean, scale, metadata = game.model_normalization()
                mean[0] = 99; scale[0] = 99; metadata['nested'][0] = 99
                self.assertEqual(network.mean[0], 1.); self.assertEqual(network.scale[0], 2.)
                self.assertEqual(network.metadata, {'nested': [3]})
                ctor.assert_called_once()
            with patch('learning.task_dagger_once_v1.policy.BoundedAgent', return_value='mock-agent') as ctor:
                self.assertEqual(game.make_agent(), 'mock-agent')
                self.assertEqual(ctor.call_args.kwargs['limits'], sources.GAME_LIMITS)


if __name__ == '__main__':
    unittest.main()
