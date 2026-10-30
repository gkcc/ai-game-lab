"""In-memory registration and verifier-restoration tests; no game or files."""
import builtins
import io
import os
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch

from . import toy_auth


def registration():
    return dict(schema=toy_auth.SCHEMA, stage_name=toy_auth.STAGE_NAME,
        engine_commit='a' * 40, engine_sha256='b' * 64,
        previous_training_result_commit='c' * 40, historical_panel_result_commit='d' * 40,
        failed_registration_commit=toy_auth.FAILED_REGISTRATION_COMMIT,
        failure_certificate_commit=toy_auth.FAILURE_CERTIFICATE_COMMIT,
        panel=dict(synthetic_seeds={'S02': 11017, 'S03': 22029},
                   active_seed_role='S02', active_opponent='market', active_side=0,
                   active_policies=['baseline', 'control'], games=2,
                   scope='public_synthetic_not_previous_panel',
                   placeholder_roles=['S03', 'auxiliary', 'round2']),
        evaluation=dict(official_games=2, official_replays=2, comparisons=1,
                        retry_allowed=False, training_updates=0, promotion=False,
                        original_24_panel_resumed=False),
        toy_model_sha256={role: digit * 64 for role, digit in
                          (('crop', '1'), ('baseline', '2'), ('control', '3'), ('auxiliary', '4'))},
        public_sources={path: 'e' * 64 for path in toy_auth.EXTRA_SOURCES | {'old/source.py'}})


def fake_identity(doc):
    module = ModuleType('synthetic_identity')
    module.REGISTRATION = 'previous-registration.json'
    module._REQUIRED = frozenset({'old/source.py'})
    module._panel = lambda _doc: None
    module.ENGINE_COMMIT, module.ENGINE_SHA = 'a' * 40, 'b' * 64
    module.RESULT_COMMIT, module.HISTORICAL_RESULT = 'c' * 40, 'd' * 40
    module.git_calls = []
    paths = toy_auth.FAILED_REGISTRATION_FILES + toy_auth.FAILURE_CERTIFICATE_FILES
    blobs = {path: ('synthetic public evidence:' + path).encode() for path in paths}
    commits = {path: toy_auth.FAILED_REGISTRATION_COMMIT for path in toy_auth.FAILED_REGISTRATION_FILES}
    commits.update({path: toy_auth.FAILURE_CERTIFICATE_COMMIT for path in toy_auth.FAILURE_CERTIFICATE_FILES})
    def git(repo, *args):
        module.git_calls.append((repo, args))
        if args[0] == 'merge-base':
            if (args[1] != '--is-ancestor' or args[2] not in set(commits.values())
                    or args[3] != 'f' * 40):
                raise ValueError('Unexpected synthetic lineage')
            return b''
        if args[0] == 'show':
            commit, path = args[1].split(':', 1)
            if commits.get(path) != commit:
                raise ValueError('Unexpected synthetic evidence commit')
            return blobs[path]
        raise ValueError('Unexpected synthetic Git command')
    module._git = git
    def file(root, path):
        if not isinstance(root, Path):
            raise TypeError('Synthetic verifier requires a Path root')
        return path
    module._file = file
    module._read = lambda path: blobs[path]
    calls = []
    def verify_public(**kwargs):
        calls.append(kwargs)
        if module.REGISTRATION != toy_auth.REGISTRATION:
            raise AssertionError('Toy registration was not installed')
        if module._REQUIRED != frozenset({'old/source.py'}) | toy_auth.EXTRA_SOURCES:
            raise AssertionError('Toy sources were not installed')
        module._panel(doc)
        return doc, {'historical': True}, {'freeze': True}
    module.verify_public = verify_public
    return module, calls


class ToyAuthTests(unittest.TestCase):
    def invoke(self, module):
        return toy_auth.verify_public(module, repo='synthetic-repo', game_root='synthetic-game',
                                      registration_commit='f' * 40)

    def assert_restored(self, module, old):
        self.assertEqual(module.REGISTRATION, old[0])
        self.assertIs(module._panel, old[1])
        self.assertIs(module._REQUIRED, old[2])

    @staticmethod
    def snapshot(module):
        return module.REGISTRATION, module._panel, module._REQUIRED

    def test_authenticates_panel_and_source_union_without_io(self):
        doc = registration()
        module, calls = fake_identity(doc)
        old = self.snapshot(module)
        with patch.object(builtins, 'open', side_effect=AssertionError('unexpected file access')), \
             patch.object(io, 'open', side_effect=AssertionError('unexpected file access')), \
             patch.object(os, 'open', side_effect=AssertionError('unexpected file access')):
            result = self.invoke(module)
        self.assertIs(result[0], doc)
        self.assertEqual(calls, [dict(repo='synthetic-repo', game_root='synthetic-game',
                                      registration_commit='f' * 40)])
        self.assertEqual(len(module.git_calls), 6)
        for commit, paths in ((toy_auth.FAILED_REGISTRATION_COMMIT, toy_auth.FAILED_REGISTRATION_FILES),
                              (toy_auth.FAILURE_CERTIFICATE_COMMIT, toy_auth.FAILURE_CERTIFICATE_FILES)):
            self.assertIn(('synthetic-repo', ('merge-base', '--is-ancestor', commit, 'f' * 40)),
                          module.git_calls)
            for path in paths:
                self.assertIn(('synthetic-repo', ('show', commit + ':' + path)), module.git_calls)
        self.assert_restored(module, old)

    def test_pathlike_repo_is_supported(self):
        doc = registration()
        module, _ = fake_identity(doc)
        old = self.snapshot(module)
        result = toy_auth.verify_public(module, repo=Path('synthetic-repo'),
                                        game_root='synthetic-game', registration_commit='f' * 40)
        self.assertIs(result[0], doc)
        self.assert_restored(module, old)

    def test_wrong_panel_types_and_identities_are_rejected_and_restored(self):
        variants = []
        for field, value in [('active_side', False), ('active_side', 0.0), ('games', 2.0),
                             ('active_seed_role', 'S03'), ('active_opponent', 'round2'),
                             ('active_policies', ['control', 'baseline']),
                             ('scope', 'independent_test')]:
            doc = registration(); doc['panel'][field] = value; variants.append(doc)
        for field, value in [('official_games', 2.0), ('official_replays', True),
                             ('comparisons', True), ('training_updates', False),
                             ('retry_allowed', 0), ('promotion', 0),
                             ('original_24_panel_resumed', 0)]:
            doc = registration(); doc['evaluation'][field] = value; variants.append(doc)
        for key, value in [('schema', 'different'), ('stage_name', 'other-stage'),
                           ('engine_commit', '0' * 40), ('engine_sha256', '0' * 64),
                           ('previous_training_result_commit', '0' * 40),
                           ('historical_panel_result_commit', '0' * 40),
                           ('failed_registration_commit', '0' * 40),
                           ('failure_certificate_commit', '0' * 40)]:
            doc = registration(); doc[key] = value; variants.append(doc)
        doc = registration(); doc['panel']['synthetic_seeds']['S02'] = 11017.0; variants.append(doc)
        doc = registration(); doc['panel']['synthetic_seeds']['S03'] = 11017; variants.append(doc)
        doc = registration(); doc['panel']['extra'] = 'not registered'; variants.append(doc)
        doc = registration(); doc['evaluation']['extra'] = False; variants.append(doc)
        for index, doc in enumerate(variants):
            module, _ = fake_identity(doc)
            old = self.snapshot(module)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.invoke(module)
            self.assert_restored(module, old)

    def test_closed_attempt_commits_and_exact_bytes_are_required(self):
        for bad_commit in (toy_auth.FAILED_REGISTRATION_COMMIT, toy_auth.FAILURE_CERTIFICATE_COMMIT):
            module, _ = fake_identity(registration())
            old, git = self.snapshot(module), module._git
            def failed_ancestor(repo, *args):
                if args[0] == 'merge-base' and args[2] == bad_commit:
                    raise ValueError('Not an ancestor')
                return git(repo, *args)
            module._git = failed_ancestor
            with self.subTest(ancestor=bad_commit), self.assertRaisesRegex(ValueError, 'Not an ancestor'):
                self.invoke(module)
            self.assert_restored(module, old)
        for changed in toy_auth.FAILED_REGISTRATION_FILES + toy_auth.FAILURE_CERTIFICATE_FILES:
            module, _ = fake_identity(registration())
            old, reader = self.snapshot(module), module._read
            module._read = lambda path: b'changed' if path == changed else reader(path)
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'evidence differs'):
                self.invoke(module)
            self.assert_restored(module, old)

    def test_every_required_source_is_enforced(self):
        for missing in toy_auth.EXTRA_SOURCES | {'old/source.py'}:
            doc = registration()
            doc['public_sources'].pop(missing)
            module, _ = fake_identity(doc)
            old = self.snapshot(module)
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'sources missing'):
                self.invoke(module)
            self.assert_restored(module, old)

    def test_model_digest_map_is_exact_and_valid(self):
        variants = []
        for key, value in [('crop', '0' * 63), ('baseline', 'G' * 64),
                           ('control', 123), ('auxiliary', 'A' * 64)]:
            doc = registration(); doc['toy_model_sha256'][key] = value; variants.append(doc)
        doc = registration(); doc['toy_model_sha256'].pop('crop'); variants.append(doc)
        doc = registration(); doc['toy_model_sha256']['extra'] = '5' * 64; variants.append(doc)
        doc = registration(); doc['toy_model_sha256']['control'] = '2' * 64; variants.append(doc)
        for index, doc in enumerate(variants):
            module, _ = fake_identity(doc)
            old = self.snapshot(module)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.invoke(module)
            self.assert_restored(module, old)

    def test_verifier_failures_always_restore_attributes(self):
        for failure in (RuntimeError('synthetic'), KeyboardInterrupt(), SystemExit(2)):
            module, _ = fake_identity(registration())
            old = self.snapshot(module)
            def failing(**_kwargs):
                module._panel(registration())
                raise failure
            module.verify_public = failing
            with self.subTest(kind=type(failure).__name__), self.assertRaises(type(failure)):
                self.invoke(module)
            self.assert_restored(module, old)

    def test_returned_registration_is_rechecked(self):
        module, _ = fake_identity(registration())
        old = self.snapshot(module)
        bad = registration(); bad['evaluation']['promotion'] = True
        module.verify_public = lambda **_kwargs: (bad, {}, {})
        with self.assertRaises(ValueError):
            self.invoke(module)
        self.assert_restored(module, old)
        module.verify_public = lambda **_kwargs: [registration(), {}, {}]
        with self.assertRaises(ValueError):
            self.invoke(module)
        self.assert_restored(module, old)

    def test_same_module_reentry_is_refused_and_restored(self):
        doc = registration()
        module, _ = fake_identity(doc)
        old = self.snapshot(module)
        def reenter(**_kwargs):
            with self.assertRaisesRegex(ValueError, 'already active'):
                self.invoke(module)
            module._panel(doc)
            return doc, {}, {}
        module.verify_public = reenter
        self.assertIs(self.invoke(module)[0], doc)
        self.assert_restored(module, old)

    def test_non_module_and_invalid_original_attributes_are_refused(self):
        with self.assertRaises(ValueError):
            self.invoke(object())
        module, _ = fake_identity(registration())
        module._REQUIRED = {'old/source.py'}
        old = self.snapshot(module)
        with self.assertRaises(ValueError):
            self.invoke(module)
        self.assert_restored(module, old)


if __name__ == '__main__':
    unittest.main()
