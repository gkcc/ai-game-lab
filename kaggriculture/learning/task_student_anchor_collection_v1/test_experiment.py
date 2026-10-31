"""Focused mixed-entry checks for empty roles, sealed results and privacy."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from learning.task_student_continuation_v1 import samples as fresh_samples
from learning.task_student_dagger_collection_v1 import planning, sampling
from learning.task_student_dagger_collection_v1.test_planning import candidate, SCHEMA, MEAN, SCALE
from . import experiment as e, merge


class EmptyRoleTests(unittest.TestCase):
    def test_role_vocabulary_is_shared_by_registration_and_merge(self):
        self.assertEqual(e.ALL_ROLES, merge.ROLES)
        self.assertEqual(e._registration()['roles'], list(merge.ROLES))

    def test_both_batches_preserve_fixed_grid_and_empty_roles(self):
        for batch in (1, 2):
            def cells():
                for role, opponent, side, window in planning.CELL_ORDER:
                    absent = batch == 2 and role in ('T03', 'T04')
                    candidates = e._empty_windows()[window] if absent else []
                    if batch == 1 and (role, opponent, side, window) == ('T03', 'market', 0, 0):
                        candidates = [candidate()]
                    yield dict(role=role, opponent=opponent, side=side,
                               window=window, candidates=candidates)
            plan = planning.create_plan(cells(), schema=SCHEMA, mean=MEAN, scale=SCALE, guard=lambda: None)
            self.assertEqual(len(plan['entries']), 144)
            if batch == 1:
                self.assertEqual(plan['summary']['selected_count'], 1)
            else:
                empty = [x for x in plan['entries'] if x['role'] in ('T03', 'T04')]
                self.assertEqual(len(empty), 48)
                for entry in empty:
                    self.assertEqual(entry['status'], 'empty')
                    self.assertIsNone(e._verify_empty_entry(entry, schema=SCHEMA, mean=MEAN,
                                                            scale=SCALE, guard=lambda: None))


class SerializationTests(unittest.TestCase):
    def test_role_round_trip_uses_contract_vocabulary(self):
        class MemoryFile(io.BytesIO):
            def __exit__(self, *unused):
                return False

        groups = [merge.Group(role, 'market', 0, 0, 0, 0,
                              np.zeros((3, 373), dtype=np.float32), 0, 1, 2)
                  for role in e.ALL_ROLES]
        output = MemoryFile()
        with patch.object(Path, 'open', return_value=output):
            digest = e._save_dataset(Path('synthetic.npz'), groups, lambda: None)
        raw = output.getvalue()
        self.assertEqual(digest, e.previous._sha(raw))
        contract = e._dataset_contract('a' * 40, 'b' * 64, digest, 'c' * 64)
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            decoded = [contract['role_vocabulary'][int(value)] for value in saved['role']]
        self.assertEqual(decoded, list(e.ALL_ROLES))
        self.assertEqual(contract['role_vocabulary'], e._registration()['roles'])
        self.assertNotEqual(e.ALL_ROLES.index('F03'), e.ALL_ROLES.index('D01'))


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.commit = 'a' * 40
        self.plan = {'schema': 'student-anchor-mixed-private-plan-v1',
                     'batch1': {'entries': [{}] * 144}, 'batch2': {'entries': [{}] * 144}}
        self.result = {'summary': {'registration_commit': self.commit, 'training': False,
                                   'independent_evaluation': False, 'status': 'failed'}}
        plan_raw, result_raw = e.previous._encoded(self.plan), e.previous._encoded(self.result)
        self.plan_sha = e.previous._sha(plan_raw)
        self.contract = {'schema': 'student-anchor-mixed-private-dataset-v1',
                         'registration_commit': self.commit, 'plan_sha256': self.plan_sha,
                         'train_sha256': 'b' * 64, 'dev_sha256': 'c' * 64,
                         'role_vocabulary': list(e.ALL_ROLES), 'model_role': 'baseline', 'training_started': False}
        contract_raw = e.previous._encoded(self.contract)
        self.files = {
            'plan.local.json': plan_raw, 'result.provisional.local.json': result_raw,
            'dataset-contract.local.json': contract_raw,
            'attempt.local.json': e.previous._encoded({'schema': 'student-anchor-mixed-attempt-v1',
                                                       'commit': self.commit, 'retry_allowed': False}),
            'plan-frozen.local.json': e.previous._encoded({'schema': 'student-anchor-mixed-plan-freeze-v1',
                      'registration_commit': self.commit, 'plan_sha256': self.plan_sha}),
            'closed.local.json': e.previous._encoded({'schema': 'student-anchor-mixed-closed-v1',
                'outcome': 'completed', 'retry_allowed': False, 'registration_commit': self.commit,
                'plan_sha256': self.plan_sha, 'result_sha256': e.previous._sha(result_raw),
                'dataset_contract_sha256': e.previous._sha(contract_raw)}),
        }
        self.failed = False

    def load(self):
        def read(path):
            name = Path(path).name
            if name not in self.files:
                raise FileNotFoundError(name)
            return self.files[name]
        with patch.object(e, 'PRIVATE', Path('synthetic-private')), \
                patch.object(e.previous, '_read', side_effect=read), \
                patch.object(Path, 'exists', side_effect=lambda: self.failed):
            return e.load_completed()

    def test_only_closed_untampered_attempt_is_accepted(self):
        self.assertEqual(self.load(), (self.result, self.contract))
        self.failed = True
        with self.assertRaises(ValueError):
            self.load()
        self.failed = False
        for name in ('plan.local.json', 'result.provisional.local.json', 'dataset-contract.local.json'):
            original = self.files[name]
            self.files[name] += b' '
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load()
            self.files[name] = original
        del self.files['closed.local.json']
        with self.assertRaises(FileNotFoundError):
            self.load()


class PriorSplitTests(unittest.TestCase):
    def test_old_development_role_cannot_enter_training_archive(self):
        arrays = dict(features=np.zeros((3, 373), dtype=np.float32),
            offsets=np.array([0, 3], dtype=np.int32), teacher=np.array([0], dtype=np.int32),
            student=np.array([1], dtype=np.int32), third=np.array([2], dtype=np.int32),
            role=np.array([2], dtype=np.int8), opponent=np.array([0], dtype=np.int8),
            side=np.array([0], dtype=np.int8), window=np.array([0], dtype=np.int8),
            step=np.array([0], dtype=np.int16), ordinal=np.array([0], dtype=np.int16))
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        raw = output.getvalue()
        ledger = {'rows': [dict(role='F03', opponent='market', side=0, window=0,
                                step=0, ordinal=0, candidate_width=3, eligible=True)]}
        roles = tuple(zip(fresh_samples.ROLES, e.FRESH_ROLES))
        with patch.object(e.previous, '_read', return_value=raw):
            with self.assertRaises(ValueError):
                e._load_prior_groups(Path('synthetic'), 'train', e.previous._sha(raw), roles, ledger)
            accepted = e._load_prior_groups(Path('synthetic'), 'dev', e.previous._sha(raw), roles, ledger)
        self.assertEqual([item.role for item in accepted], ['F03'])


class CliTests(unittest.TestCase):
    def test_bad_arguments_and_exceptions_do_not_echo_private_text(self):
        secret = 'synthetic_private_location'
        for args in ([secret], ['prepare', secret], ['--' + secret]):
            output = io.StringIO()
            with self.subTest(args=args), patch.object(sys, 'argv', ['experiment', *args]), redirect_stderr(output):
                with self.assertRaises(SystemExit) as caught:
                    e.cli()
            self.assertEqual(caught.exception.code, 1)
            self.assertNotIn(secret, output.getvalue())
        output = io.StringIO()
        with patch.object(e, 'main', side_effect=FileNotFoundError(secret)), redirect_stderr(output):
            with self.assertRaises(SystemExit) as caught:
                e.cli()
        self.assertEqual(caught.exception.code, 1)
        self.assertNotIn(secret, output.getvalue())


if __name__ == '__main__':
    unittest.main()
