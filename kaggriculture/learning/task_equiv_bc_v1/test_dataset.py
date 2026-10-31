"""Synthetic in-memory checks; no private source, model or game is opened."""
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import io
import json
import unittest

from decision.api import observe
from learning.bc_crop.agent import TEACHER_SHA
from learning.bc_crop.experiment import ENGINE_COMMIT, ENGINE_SHA
from learning.task_choice_v1.dataset import canonical, digest
from learning.task_choice_v1.features import SCHEMA_VERSION, candidate_id, schema_hash
from . import dataset


def raw(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def evidence():
    train, validation = list(range(8)), [100, 101]
    schema = schema_hash([f'local{i}' for i in range(98)], [f'shared{i}' for i in range(275)])
    cfg = dict(experiment='joint-task-choice-interface-v1', demonstrations=60,
               training_runs=0, new_complete_games=0, engine_commit=ENGINE_COMMIT,
               engine_sha256=ENGINE_SHA, teacher_sha256=TEACHER_SHA,
               train_seeds=train, validation_seeds=validation,
               source_ledgers={'collect-ledger.json': 'a'*64, 'collect-smoke-ledger.json': 'b'*64})
    rows, files = [], []
    for seed in train + validation:
        split = 'train' if seed in train else 'validation'
        for opponent in dataset.ALL_OPPONENTS:
            for side in (0, 1):
                game = f'game-{seed}-{opponent}-{side}'
                streams = {k: dict(path=f'data/{split}/{game}.{k}.jsonl.gz',
                                   sha256='1'*64, rows=6) for k in ('inputs', 'labels', 'index')}
                audit = dict(passed=True, official_transitions_reexecuted=719,
                             both_public_and_private_states_match_each_step=True,
                             inventory_order_checked_each_step=True, engine_sha256=ENGINE_SHA)
                rows.append(dict(game=game, seed=seed, opponent=opponent, side=side, split=split,
                    passed=True, reconstructed_actor_actions=719, original_teacher_actions_checked=719,
                    reproduced_rival_actions=719, replay_gzip_sha256='2'*64,
                    schema_version=SCHEMA_VERSION, schema_sha256=schema, decisions=6,
                    audit=audit, files=streams))
                files.append(dict(game=game, split=split, files=deepcopy(streams)))
    ledger = dict(passed=True, games=60, games_dispatched=60, failures=[], new_games=0,
                  training_runs=0, official_existing_replay_transitions=60*719, rows=rows)
    cfg_raw, ledger_raw = raw(cfg), raw(ledger)
    manifest = dict(registration_sha256=sha(cfg_raw), reconstruction_ledger_sha256=sha(ledger_raw),
                    teacher_sha256=TEACHER_SHA, schema_version=SCHEMA_VERSION, schema_sha256=schema,
                    source_ledgers=deepcopy(cfg['source_ledgers']), training_runs=0, new_games=0,
                    examples=360, files=files)
    manifest_raw = raw(manifest)
    review = dict(passed=True, all_dataset_files_hashed=180, all_dataset_rows_aligned=360,
                  ledger_sha256=sha(ledger_raw))
    review_raw = raw(review)
    source_schema = dict(schema_version=SCHEMA_VERSION, schema_sha256=schema,
                         feature_names=[f'local{i}' for i in range(98)],
                         shared_feature_names=[f'shared{i}' for i in range(275)],
                         shared_dimension=275, candidate_dimension=98, labels_separate=True,
                         metadata_seed_game_separate=True, inventory_order_explicit=True)
    schema_raw = raw(source_schema)
    acceptance = dict(stage_execution_passed=True, structural_coverage_passed=True,
                      teacher_mode_full_action_equivalence_passed=True,
                      original_demonstrations_reconstructed=60, all_dataset_rows_independently_consumed=True,
                      dataset_files=180, task_examples=360,
                      evidence_sha256={'registration.json': sha(cfg_raw),
                          'dataset-manifest.json': sha(manifest_raw),
                          'reconstruction-ledger.json': sha(ledger_raw),
                          'independent-review.json': sha(review_raw), 'schema.json': sha(schema_raw)})
    acceptance_raw = raw(acceptance)
    return dict(registration_raw=cfg_raw, manifest_raw=manifest_raw, ledger_raw=ledger_raw,
                review_raw=review_raw, schema_raw=schema_raw, acceptance_raw=acceptance_raw,
                expected_acceptance_sha256=sha(acceptance_raw), excluded_seeds=[0, 100])


def budget():
    return dataset.ReadBudget(compressed_limit=16*1024*1024, decoded_limit=128*1024*1024,
                              check=lambda: True)


def make_replay(seed=1, opponent_private='OPPONENT_SECRET'):
    farms = [dict(money=1000, tiles=[[None]], farmer=[0, 0], hands=[[0, 0]],
                  unlocked_quadrants=[], hires_today=0, private={'leak': opponent_private})
             for _ in range(2)]
    private = dict(shed={'WHEAT': 3, 'CARROT': 2}, seeds={'WHEAT': 2},
                   inventories=[{'WHEAT': 2, 'CARROT': 1}, {}], secret='FIRST_ACTOR_SECRET')
    other_private = dict(shed={'MILK': 4}, seeds={'MELON': 3},
                         inventories=[{'MILK': 1}, {}], secret=opponent_private)
    cfg = {'boardSize': 1, 'marketParams': {}}
    steps = []
    for step in range(720):
        public = dict(step=step, day=1, hour=0, farms=deepcopy(farms),
                      market={'inventory': {'WHEAT': 10}, 'prices': {}, 'params': {}},
                      town={'unlocked_shops': []})
        frame = []
        for side in (0, 1):
            obs = deepcopy(public)
            obs.update(player=side, private=deepcopy(private if side == 0 else other_private))
            frame.append(dict(observation=obs, status='DONE' if step == 719 else 'ACTIVE',
                              action={'farmer': ['PASS'], 'hands': [['PASS']], 'market': []}))
        steps.append(frame)
    return dict(info={'seed': seed, 'secret': 'DO_NOT_SEND'}, configuration=cfg, steps=steps)


def make_input(replay, step, count=2, side=0):
    view = observe(replay['steps'][step][side]['observation'], replay['configuration'])
    local = [f'local{i}' for i in range(98)]
    shared = [f'shared{i}' for i in range(275)]
    candidates = [dict(id=candidate_id(i, (0, 0), 'WATER'), unit=i, target=[0, 0],
                       op='WATER', x=[0.0]*98) for i in range(count)]
    return dict(schema_version=SCHEMA_VERSION, schema_sha256=schema_hash(local, shared),
        feature_names=local, shared_feature_names=shared, shared_x=[0.0]*275,
        candidate_scope='frozen teacher economic opportunities; not official legality masks',
        observation=view.observation, configuration=view.configuration,
        planned=dict(seeds={}, needed_seeds={}, supply={}, projected_shed={}, unit_actions=[],
                     available_units=[], interpretation='before assignment: reservations and requests, not official executed state'),
        candidates=candidates, inventory_item_order=[list(x) for x in view.private['inventories']],
        shed_item_order=list(view.private['shed']))


def make_rows(replay, side=0):
    rows = []
    for step in (0, 240, 480):
        for decision, count in ((0, 1), (1, 257), (2, 2), (3, 2)):
            inputs = make_input(replay, step, count, side)
            label = {'teacher_label': inputs['candidates'][0]['id'], 'index': 0}
            index = dict(game='fixture', seed=1, opponent='market', side=side, step=step,
                         decision=decision, split='train', input_sha256=digest(inputs))
            rows.append((inputs, label, index))
    return rows


def streams(rows, replay):
    data = {}
    for index, kind in enumerate(('inputs', 'labels', 'index')):
        uncompressed = ''.join(canonical(row[index]) + '\n' for row in rows).encode()
        data[kind] = gzip.compress(uncompressed, mtime=0)
    data['replay'] = gzip.compress(raw(replay), mtime=0)
    files = tuple((k, dataset.FileBinding(f'data/train/fixture.{k}.jsonl.gz',
                                         sha(data[k]), len(rows)))
                  for k in ('inputs', 'labels', 'index'))
    binding = dataset.GameBinding('fixture', 1, 'market', rows[0][2]['side'], 'train', 'train',
                                  sha(data['replay']), rows[0][0]['schema_sha256'], files)
    return binding, data


def sample(binding, data, meter=None):
    return dataset.sample_game(binding, inputs_stream=io.BytesIO(data['inputs']),
        labels_stream=io.BytesIO(data['labels']), index_stream=io.BytesIO(data['index']),
        replay_stream=io.BytesIO(data['replay']), budget=meter or budget())


class BindingTests(unittest.TestCase):
    def test_fixed_seed_selection_and_exact_twelve_game_plan(self):
        plan = dataset.bind_sources(**evidence())
        self.assertEqual(len(plan), 12)
        self.assertEqual([(p.seed, p.split) for p in plan[::4]], [(1, 'train'), (2, 'train'), (101, 'dev')])
        self.assertEqual([(p.opponent, p.side) for p in plan[:4]],
                         [('market', 0), ('market', 1), ('round2', 0), ('round2', 1)])

    def test_reject_unanchored_acceptance_and_changed_old_review(self):
        args = evidence()
        args['expected_acceptance_sha256'] = 'f'*64
        with self.assertRaises(ValueError): dataset.bind_sources(**args)
        args = evidence()
        args['review_raw'] = raw({'passed': True})
        with self.assertRaises(ValueError): dataset.bind_sources(**args)

    def test_reject_manifest_identity_and_duplicate_population(self):
        args = evidence()
        manifest = json.loads(args['manifest_raw'])
        manifest['files'][1]['game'] = manifest['files'][0]['game']
        args['manifest_raw'] = raw(manifest)
        acceptance = json.loads(args['acceptance_raw'])
        acceptance['evidence_sha256']['dataset-manifest.json'] = sha(args['manifest_raw'])
        args['acceptance_raw'] = raw(acceptance)
        args['expected_acceptance_sha256'] = sha(args['acceptance_raw'])
        with self.assertRaises(ValueError): dataset.bind_sources(**args)

    def test_reject_no_disjoint_validation_and_wrong_exclusion_count(self):
        args = evidence()
        args['excluded_seeds'] = [100, 101]
        with self.assertRaises(ValueError): dataset.bind_sources(**args)
        args['excluded_seeds'] = [100]
        with self.assertRaises(ValueError): dataset.bind_sources(**args)


class SampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = make_replay()
        cls.rows = make_rows(cls.replay)

    def test_selection_prefix_order_and_private_observation(self):
        binding, data = streams(self.rows, self.replay)
        meter = budget()
        chosen = sample(binding, data, meter)
        self.assertEqual([(x['step'], x['decision']) for x in chosen],
                         [(0,2), (0,3), (240,2), (240,3), (480,2), (480,3)])
        self.assertEqual([len(x['teacher_prefix']) for x in chosen], [2,3,2,3,2,3])
        self.assertEqual(list(chosen[0]['inputs']['observation']['private']['shed']), ['WHEAT', 'CARROT'])
        self.assertEqual(list(chosen[0]['inputs']['observation']['private']['inventories'][0]),
                         ['WHEAT', 'CARROT'])
        self.assertNotIn('OPPONENT_SECRET', json.dumps(chosen))
        self.assertNotIn('DO_NOT_SEND', json.dumps(chosen))
        self.assertEqual(meter.parsed_groups, len(self.rows))

    def test_opponent_private_changes_cannot_enter_actor_input(self):
        changed = make_replay(opponent_private='DIFFERENT_OPPONENT_SECRET')
        binding, data = streams(self.rows, changed)
        chosen = sample(binding, data)
        self.assertNotIn('DIFFERENT_OPPONENT_SECRET', json.dumps(chosen))

    def test_second_seat_uses_only_its_own_private_observation(self):
        rows = make_rows(self.replay, side=1)
        binding, data = streams(rows, self.replay)
        chosen = sample(binding, data)
        observed = chosen[0]['inputs']['observation']
        self.assertEqual(observed['player'], 1)
        self.assertEqual(observed['private']['shed'], {'MILK': 4})
        self.assertNotIn('FIRST_ACTOR_SECRET', json.dumps(chosen))
        self.assertNotIn('OPPONENT_SECRET', json.dumps(chosen))

    def test_short_or_extra_stream_and_bad_source_sha_fail(self):
        binding, data = streams(self.rows, self.replay)
        broken = dict(data)
        broken['labels'] = gzip.compress(b'{}\n', mtime=0)
        with self.assertRaises(ValueError): sample(binding, broken)
        broken = dict(data)
        broken['inputs'] += b'x'
        with self.assertRaises((ValueError, gzip.BadGzipFile, EOFError)): sample(binding, broken)

    def test_line_count_mismatch_even_with_matching_compressed_hash(self):
        binding, data = streams(self.rows, self.replay)
        # Make the shorter stream's stored SHA internally consistent, so the
        # failure must come from stream alignment rather than a stale digest.
        data['labels'] = gzip.compress(
            ''.join(canonical(row[1]) + '\n' for row in self.rows[:-1]).encode(), mtime=0)
        revised = tuple((kind, replace(spec, sha256=sha(data[kind])))
                        if kind == 'labels' else (kind, spec) for kind, spec in binding.files)
        binding = replace(binding, files=revised)
        meter = budget()
        with self.assertRaisesRegex(ValueError, 'Unaligned stream lengths'):
            sample(binding, data, meter)
        self.assertEqual(meter.stream_lines, {'inputs': 12, 'labels': 11, 'index': 12})
        self.assertEqual((meter.replay_json_attempts, meter.started_groups, meter.parsed_groups), (1, 12, 11))

    def test_wrong_input_digest_or_wrong_identity_fails(self):
        for mutate in ('digest', 'side', 'seed', 'opponent'):
            rows = deepcopy(self.rows)
            if mutate == 'digest': rows[0][2]['input_sha256'] = 'f'*64
            elif mutate == 'side': rows[0][2]['side'] = 1
            elif mutate == 'seed': rows[0][2]['seed'] = True
            else: rows[0][2]['opponent'] = 'round2'
            binding, data = streams(rows, self.replay)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): sample(binding, data)

    def test_wrong_label_or_sequence_fails(self):
        for mutate in ('index', 'label', 'order', 'gap'):
            rows = deepcopy(self.rows)
            if mutate == 'index': rows[0][1]['index'] = 99
            elif mutate == 'label': rows[0][1]['teacher_label'] = 'wrong'
            elif mutate == 'order': rows[1], rows[2] = rows[2], rows[1]
            else: rows[1][2]['decision'] = 3
            binding, data = streams(rows, self.replay)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): sample(binding, data)

    def test_missing_window_fails_without_substitution(self):
        rows = [row for row in self.rows if row[2]['step'] != 480]
        binding, data = streams(rows, self.replay)
        with self.assertRaises(ValueError): sample(binding, data)

    def test_opponent_private_or_seed_in_stored_model_input_is_rejected(self):
        for mutate in ('opponent_private', 'seed'):
            rows = deepcopy(self.rows)
            if mutate == 'opponent_private':
                rows[0][0]['observation']['farms'][1]['private'] = {'hidden': 1}
            else:
                rows[0][0]['seed'] = 1
            rows[0][2]['input_sha256'] = digest(rows[0][0])
            binding, data = streams(rows, self.replay)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): sample(binding, data)

    def test_inventory_order_metadata_mismatch_fails(self):
        rows = deepcopy(self.rows)
        rows[0][0]['inventory_item_order'][0] = ['WHEAT', 'WHEAT']
        rows[0][2]['input_sha256'] = digest(rows[0][0])
        binding, data = streams(rows, self.replay)
        with self.assertRaises(ValueError): sample(binding, data)

    def test_bad_replay_seed_or_insufficient_budget_fails(self):
        changed = deepcopy(self.replay)
        changed['info']['seed'] = 2
        binding, data = streams(self.rows, changed)
        with self.assertRaises(ValueError): sample(binding, data)
        binding, data = streams(self.rows, self.replay)
        with self.assertRaises(ValueError):
            sample(binding, data, dataset.ReadBudget(compressed_limit=10, decoded_limit=100, check=lambda: True))


if __name__ == '__main__':
    unittest.main()
