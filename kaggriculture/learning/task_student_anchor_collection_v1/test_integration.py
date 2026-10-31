"""In-memory wiring check for both Anchor passes; no actual game or model."""
from contextlib import ExitStack, redirect_stdout
from collections import Counter
import hashlib
import io
import json
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import decision
import decision.api
import learning.task_student_continuation_v1 as continuation
from learning.task_student_continuation_v1 import experiment as previous
from learning.task_student_dagger_collection_v1 import dataset
from learning.task_student_dagger_collection_v1.test_planning import candidate, SCHEMA, MEAN, SCALE
from learning.task_student_anchor_collection_v1 import experiment, merge, sources
from learning.task_student_anchor_collection_v1 import test_sources as source_fixture


class IntegrationTests(unittest.TestCase):
    def test_two_batches_freeze_before_second_pass_with_virtual_slots(self):
        # This fixture replaces every private source read and old binding
        # verifier with synthetic memory-backed data; it opens no real replay.
        fixture = source_fixture.SourceTests('test_two_pass_limit_requires_normal_semantic_completion')
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        context = fixture.context()
        manifest = {name: {'train_sha256': 'a'*64, 'dev_sha256': 'b'*64}
                    for name in ('fresh', 'dagger')}
        binding = dict(prior=manifest, anchor_roles=context.private_role_binding(),
                       task_sha256=context.task_sha256, crop_sha256=context.crop_sha256)
        files, coordinates, scans, comparisons, transitions, saves = {}, [], [], [], [], []
        visits = Counter()

        def write(path, value):
            key = str(path)
            if key in files:
                raise AssertionError('Synthetic write collision')
            files[key] = previous._encoded(value)

        def read(path):
            return files[str(path)]

        source_read = sources.read_game
        def read_game(ctx, *, batch, role, opponent, side, guard):
            key = (batch, role, opponent, side)
            visit = visits[key]
            visits[key] += 1
            if visit == 1:
                plan_key = str(experiment.PRIVATE/'plan.local.json')
                freeze_key = str(experiment.PRIVATE/'plan-frozen.local.json')
                self.assertTrue(plan_key in files and freeze_key in files, 'Both plans must be frozen')
                plan, frozen = json.loads(files[plan_key]), json.loads(files[freeze_key])
                self.assertEqual(frozen['plan_sha256'], previous._sha(files[plan_key]))
                self.assertEqual([len(plan['batch'+str(i)]['entries']) for i in (1, 2)], [144, 144])
            self.assertFalse(batch == 2 and role in ('T03', 'T04'))
            coordinates.append((*key, visit))
            return source_read(ctx, batch=batch, role=role, opponent=opponent, side=side, guard=guard)

        source_scan = sources.PrivateGame.scan_candidates
        def scan(game, *, guard):
            # Exercise the actual 719-line synthetic scan and lifecycle, then
            # inject a well-formed ranked candidate in one game per batch to
            # test the selected branch without loading any real model.
            accounting, _ = source_scan(game, guard=guard)
            identity = game.identity
            key = (identity['batch'], identity['role'], identity['opponent'], identity['side'])
            scans.append(key)
            windows = [[] for _ in range(6)]
            if key in ((1, 'T01', 'market', 0), (2, 'T01', 'market', 0)):
                windows[0] = [candidate(0, offset=1, value=100.0+identity['batch'])]
            return accounting, windows

        class Outcome:
            reference_action = {'farmer': ['PASS'], 'hands': [], 'market': []}
            alternative_action = reference_action
            def public_summary(self):
                return {'status': 'same', 'checks': {}}

        def compare_turn(*_args, **kwargs):
            self.assertTrue(str(experiment.PRIVATE/'plan-frozen.local.json') in files, 'Plan must be frozen')
            comparisons.append(kwargs['alternative'])
            return Outcome()

        def compare_step(*_args, **_kwargs):
            transitions.append(True)
            return {'factual_reproduced': True}

        fake_compare = types.ModuleType('learning.task_student_continuation_v1.compare')
        fake_compare.compare_turn = compare_turn
        fake_official = types.ModuleType('learning.task_student_continuation_v1.official_step')
        fake_official.compare_step = compare_step
        fake_observation = types.ModuleType('decision.audit_market_teacher')
        fake_observation.observation = lambda frame, side: dict(
            player=side, step=frame[0]['observation']['step'])
        fake_modules = {fake_compare.__name__: fake_compare, fake_official.__name__: fake_official,
                        fake_observation.__name__: fake_observation}

        def load_groups(directory, split, *_args):
            source = 'fresh' if directory == experiment.fresh_collection.PRIVATE else 'dagger'
            size = {('fresh','train'): 7, ('fresh','dev'): 2,
                    ('dagger','train'): 15, ('dagger','dev'): 4}[(source, split)]
            return [object() for _ in range(size)]

        def save(_path, _groups, _guard):
            saves.append(True)
            return hashlib.sha256(str(len(saves)).encode()).hexdigest()

        def eligible_example(**_kwargs):
            return types.SimpleNamespace(features=np.zeros((3, 373), dtype=np.float32),
                teacher_index=0, student_index=1, third_index=2)

        def merge_groups(groups, *, mean, scale):
            self.assertEqual(len(groups), 30)
            self.assertEqual([group.role for group in groups[-2:]], ['A01', 'A05'])
            return (), (), {'status': 'failed'}

        expected_mapping = experiment._registration()['anchor_batches']
        for batch in (1, 2):
            for role in sources.ROLES:
                self.assertEqual(context.source_role(batch=batch, role=role), expected_mapping[batch-1][role])

        with ExitStack() as stack:
            patches = (
                patch.dict(sys.modules, fake_modules),
                patch.dict(experiment.os.environ, {}, clear=False),
                patch.object(continuation, 'compare', fake_compare, create=True),
                patch.object(continuation, 'official_step', fake_official, create=True),
                patch.object(decision, 'audit_market_teacher', fake_observation, create=True),
                patch.object(previous, 'physical_available', return_value=99*1024**3),
                patch.object(previous, '_guard', return_value=None),
                patch.object(previous, '_write_new', side_effect=write),
                patch.object(previous, '_read', side_effect=read),
                patch.object(experiment.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=99*1024**3)),
                patch.object(experiment, '_runtime_binding', return_value=('a'*40, binding)),
                patch.object(experiment, '_prior_snapshot', return_value=(manifest, {}, {})),
                patch.object(experiment, '_load_prior_groups', side_effect=load_groups),
                patch.object(experiment, '_save_dataset', side_effect=save),
                patch.object(sources, 'open_context', return_value=context),
                patch.object(sources.PrivateContext, 'model_normalization', return_value=(MEAN, SCALE, SCHEMA)),
                patch.object(sources, 'read_game', side_effect=read_game),
                patch.object(sources.PrivateGame, 'scan_candidates', scan),
                patch.object(decision.api, 'observe', side_effect=lambda observation, configuration:
                             types.SimpleNamespace(observation=observation, configuration={})),
                patch.object(dataset, 'make_example', side_effect=eligible_example),
                patch.object(merge, 'merge', side_effect=merge_groups),
                redirect_stdout(io.StringIO()),
            )
            for item in patches:
                stack.enter_context(item)
            experiment.run()

        expected = {(batch, role, opponent, side) for batch, roles in sources.SLOTS.items()
                    for role in roles for opponent in sources.OPPONENTS for side in (0, 1)}
        first = {(batch, role, opponent, side) for batch, role, opponent, side, visit in coordinates if visit == 0}
        second = {(batch, role, opponent, side) for batch, role, opponent, side, visit in coordinates if visit == 1}
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual((len(coordinates), len(scans)), (80, 80))
        plan = json.loads(files[str(experiment.PRIVATE/'plan.local.json')])
        empty = [entry for entry in plan['batch2']['entries'] if entry['role'] in ('T03', 'T04')]
        self.assertEqual(len(empty), 48)
        self.assertTrue(all(entry['status'] == 'empty' for entry in empty))
        self.assertEqual((len(comparisons), len(transitions), len(saves)), (4, 4, 2))
        result = json.loads(files[str(experiment.PRIVATE/'result.provisional.local.json')])
        self.assertEqual(len(result['rows']), 240)
        self.assertEqual((result['summary']['selected'], result['summary']['anchor_eligible']), (2, 2))
        self.assertTrue(str(experiment.PRIVATE/'closed.local.json') in files, 'Completion marker missing')


if __name__ == '__main__':
    unittest.main()
