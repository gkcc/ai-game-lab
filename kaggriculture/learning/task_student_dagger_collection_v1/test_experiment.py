"""Regress completion sealing and accidental private-path disclosure."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from . import experiment as e


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.private = Path('synthetic-private')
        self.commit = 'a' * 40
        self.plan = {'schema': 'student-dagger-private-plan-v1', 'entries': [{}] * 144}
        self.result = {'summary': {'registration_commit': self.commit, 'training': False,
                                   'independent_evaluation': False, 'status': 'failed'}}
        self.plan_raw, self.result_raw = e.previous._encoded(self.plan), e.previous._encoded(self.result)
        self.plan_sha = e.previous._sha(self.plan_raw)
        self.contract = {'schema': 'student-dagger-private-dataset-v1', 'registration_commit': self.commit,
                         'plan_sha256': self.plan_sha, 'train_sha256': 'b' * 64, 'dev_sha256': 'c' * 64,
                         'model_role': 'baseline', 'training_started': False}
        self.contract_raw = e.previous._encoded(self.contract)
        self.files = {
            'plan.local.json': self.plan_raw,
            'result.provisional.local.json': self.result_raw,
            'dataset-contract.local.json': self.contract_raw,
            'attempt.local.json': e.previous._encoded({'schema': 'student-dagger-attempt-v1',
                                                       'commit': self.commit, 'retry_allowed': False}),
            'plan-frozen.local.json': e.previous._encoded({'schema': 'student-dagger-plan-freeze-v1',
                    'registration_commit': self.commit, 'plan_sha256': self.plan_sha}),
            'closed.local.json': e.previous._encoded({'schema': 'student-dagger-closed-v1',
                'outcome': 'completed', 'retry_allowed': False, 'registration_commit': self.commit,
                'plan_sha256': self.plan_sha, 'result_sha256': e.previous._sha(self.result_raw),
                'dataset_contract_sha256': e.previous._sha(self.contract_raw)}),
        }
        self.failed = False

    def load(self):
        def read(path):
            name = Path(path).name
            if name not in self.files:
                raise FileNotFoundError(name)
            return self.files[name]
        with patch.object(e, 'PRIVATE', self.private), patch.object(e.previous, '_read', side_effect=read), \
                patch.object(Path, 'exists', side_effect=lambda: self.failed):
            return e.load_completed()

    def test_only_sealed_result_is_accepted(self):
        result, contract = self.load()
        self.assertEqual(result, self.result)
        self.assertEqual(contract, self.contract)
        self.failed = True
        with self.assertRaises(ValueError):
            self.load()
        self.failed = False
        del self.files['closed.local.json']
        with self.assertRaises(FileNotFoundError):
            self.load()

    def test_tampering_any_bound_input_is_rejected(self):
        for name in ('plan.local.json', 'result.provisional.local.json', 'dataset-contract.local.json',
                     'attempt.local.json', 'plan-frozen.local.json'):
            original = self.files[name]
            self.files[name] = original + b' '
            if name in ('attempt.local.json', 'plan-frozen.local.json'):
                # Whitespace is semantically harmless for these two documents;
                # replace a required value to test their actual contract.
                value = e.previous._document(original)
                value['retry_allowed' if name.startswith('attempt') else 'plan_sha256'] = True
                self.files[name] = e.previous._encoded(value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load()
            self.files[name] = original


class CliTests(unittest.TestCase):
    def test_path_bearing_exception_is_redacted(self):
        secret = 'synthetic_private_location'
        output = io.StringIO()
        with patch.object(e, 'main', side_effect=FileNotFoundError(secret)), redirect_stderr(output):
            with self.assertRaises(SystemExit) as caught:
                e.cli()
        self.assertEqual(caught.exception.code, 1)
        self.assertNotIn(secret, output.getvalue())
        self.assertIn('"category": "io"', output.getvalue())

    def test_invalid_arguments_are_not_echoed(self):
        secret = 'synthetic_private_location'
        for args in ([secret], ['prepare', secret], ['--' + secret]):
            output = io.StringIO()
            with self.subTest(args=args), patch.object(sys, 'argv', ['experiment', *args]), redirect_stderr(output):
                with self.assertRaises(SystemExit) as caught:
                    e.cli()
            self.assertEqual(caught.exception.code, 1)
            self.assertNotIn(secret, output.getvalue())
            self.assertIn('"category": "invalid_or_runtime"', output.getvalue())


if __name__ == '__main__':
    unittest.main()
