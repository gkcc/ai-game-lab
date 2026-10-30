"""Synthetic registration/identity failure cases; no filesystem or network IO."""
import json
import hashlib
import io
import subprocess
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import identity


def registered():
    return dict(schema='task-pair-dev-games-prereg-v1',
        previous_training_result_commit=identity.RESULT_COMMIT,
        historical_panel_result_commit=identity.HISTORICAL_RESULT,
        engine_commit=identity.ENGINE_COMMIT, engine_sha256=identity.ENGINE_SHA,
        panel=dict(seed_roles=['S02', 'S03'], opponents=['market', 'round2'],
                   policies=['baseline', 'control', 'auxiliary'], sides=[0, 1], games=24,
                   scope='consumed_development_no_promotion'),
        evaluation=dict(comparisons=['control_minus_baseline', 'auxiliary_minus_control'],
            cluster='seed', min_per_seed=5, min_total=20, significance='not_assessed',
            promotion=False, retries=False, official_games=24, official_replays=24,
            training_updates=0, new_seed_generation=False))


class RegistrationIdentity(unittest.TestCase):
    def test_predeclared_consumed_panel(self):
        identity._panel(registered())
        for key, bad in [('seed_roles', ['S01', 'S03']), ('sides', [False, True]),
                         ('scope', 'holdout'), ('games', 25)]:
            doc = registered()
            doc['panel'][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                identity._panel(doc)

    def test_engine_and_previous_result_are_fixed(self):
        for key in ('engine_commit', 'engine_sha256', 'previous_training_result_commit',
                    'historical_panel_result_commit'):
            doc = registered()
            doc[key] = '0'*len(doc[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                identity._panel(doc)

    def test_threshold_or_scope_cannot_be_relabeled(self):
        for key, bad in [('min_total', 19), ('official_games', 23),
                         ('training_updates', False), ('promotion', 0)]:
            doc = registered()
            doc['evaluation'][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                identity._panel(doc)

    def test_duplicate_or_nonfinite_registration_is_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'[]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                identity._json(raw)
        self.assertEqual(identity._json(json.dumps(registered()).encode()), registered())

    def test_source_paths_cannot_escape(self):
        for path in ('../a.py', '/a.py', 'a/../b.py', 'a\\b.py', 'C' + ':' + '/a.py', 'a//b.py'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                identity._relative(path)
        self.assertEqual(identity._relative('a/b.py'), 'a/b.py')

    def test_empty_frozen_source_is_hashed_but_empty_document_is_rejected(self):
        with patch.object(identity.Path, 'open', side_effect=lambda *_a, **_k: io.BytesIO(b'')):
            self.assertEqual(identity._read('synthetic', allow_empty=True), b'')
            with self.assertRaisesRegex(ValueError, 'Source size limit'):
                identity._read('synthetic')
            with patch.object(identity, '_file', return_value='synthetic'):
                identity._check_digest('root', 'a/empty.py', hashlib.sha256(b'').hexdigest())
                with self.assertRaisesRegex(ValueError, 'Source bytes changed'):
                    identity._check_digest('root', 'a/empty.py', '0'*64)

    def test_git_failure_does_not_disclose_captured_error(self):
        result = SimpleNamespace(returncode=1, stdout=b'', stderr=b'synthetic-private-marker')
        with patch.object(identity.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(ValueError, '^Git verification failed$') as error:
                identity._git('synthetic-repo', 'status')
        self.assertNotIn('synthetic-private-marker', str(error.exception))

    def test_git_timeout_or_startup_failure_does_not_disclose_details(self):
        marker = 'synthetic-private-marker'
        errors = (subprocess.TimeoutExpired([marker], 1, output=marker.encode()),
                  FileNotFoundError(marker))
        for failure in errors:
            with self.subTest(kind=type(failure).__name__), \
                 patch.object(identity.subprocess, 'run', side_effect=failure):
                with self.assertRaisesRegex(ValueError, '^Git verification failed$') as error:
                    identity._git('synthetic-repo', 'status')
                formatted = ''.join(traceback.format_exception(error.exception))
                self.assertNotIn(marker, formatted)

    def test_historical_evidence_is_bound_to_its_original_commits(self):
        calls = []
        def git(_repo, *args):
            calls.append(args)
            return b'' if args[0] == 'merge-base' else b'original'
        with patch.object(identity, '_git', side_effect=git), \
             patch.object(identity, '_file', side_effect=lambda _repo, rel: rel), \
             patch.object(identity, '_read', return_value=b'original'):
            identity._historical_lineage('repo', 'a'*40)
        self.assertIn(('merge-base', '--is-ancestor', identity.HISTORICAL_RESULT, 'a'*40), calls)
        self.assertIn(('merge-base', '--is-ancestor', identity.RESULT_COMMIT, 'a'*40), calls)
        self.assertTrue(any(args == ('show', identity.HISTORICAL_RESULT + ':' + identity.HISTORICAL_FREEZE)
                            for args in calls))
        self.assertTrue(any(args == ('show', identity.RESULT_COMMIT + ':' +
                            'review/evidence/task-pair-wsl-training-v1-result-20261030/certificate.json')
                            for args in calls))
        with patch.object(identity, '_git', side_effect=git), \
             patch.object(identity, '_file', side_effect=lambda _repo, rel: rel), \
             patch.object(identity, '_read', return_value=b'changed'):
            with self.assertRaisesRegex(ValueError, 'Historical evidence bytes changed'):
                identity._historical_lineage('repo', 'a'*40)


if __name__ == '__main__':
    unittest.main()
