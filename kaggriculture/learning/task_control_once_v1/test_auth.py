"""Synthetic checks for single-game scope and verifier restoration."""
import copy
import hashlib
from types import ModuleType, SimpleNamespace
import unittest

from . import auth


def identity():
    return SimpleNamespace(ENGINE_COMMIT='engine', ENGINE_SHA='rules', RESULT_COMMIT='training',
                           HISTORICAL_RESULT='historical')


def registration():
    return dict(schema=auth.SCHEMA, stage_name=auth.STAGE_NAME, panel=copy.deepcopy(auth.PANEL),
                evaluation=copy.deepcopy(auth.EVALUATION), engine_commit='engine', engine_sha256='rules',
                previous_training_result_commit='training', historical_panel_result_commit='historical',
                public_sources={'required': 'placeholder'})


class AuthTests(unittest.TestCase):
    def test_scope_is_exact_and_bool_is_not_an_integer(self):
        reg = registration()
        auth._panel(reg, identity(), {'required'})
        mutations = []
        change = copy.deepcopy(reg); change['panel']['games'] = True; mutations.append(change)
        change = copy.deepcopy(reg); change['panel']['actual_seed_role'] = 'S02'; mutations.append(change)
        change = copy.deepcopy(reg); change['panel']['actual_opponent'] = 'market'; mutations.append(change)
        change = copy.deepcopy(reg); change['evaluation']['official_games'] = 2; mutations.append(change)
        change = copy.deepcopy(reg); change['evaluation']['retries'] = True; mutations.append(change)
        change = copy.deepcopy(reg); change['engine_sha256'] = 'other'; mutations.append(change)
        for bad in mutations:
            with self.subTest(), self.assertRaises(ValueError):
                auth._panel(bad, identity(), {'required'})

    def test_isolated_verifier_state_restored_after_failure(self):
        module = ModuleType('synthetic_identity')
        module.REGISTRATION = 'old-registration'
        module._panel = lambda _value: None
        module._REQUIRED = frozenset({'old-source'})
        originals = module.REGISTRATION, module._panel, module._REQUIRED
        def fail(**_kwargs):
            self.assertEqual(module.REGISTRATION, auth.REGISTRATION)
            self.assertTrue(auth.EXTRA_SOURCES.issubset(module._REQUIRED))
            raise ValueError('synthetic stop')
        module.verify_public = fail
        with self.assertRaises(ValueError):
            auth.verify_public(module, repo='unused', game_root='unused', registration_commit='unused')
        self.assertEqual((module.REGISTRATION, module._panel, module._REQUIRED), originals)

    def test_new_registration_cannot_replace_historical_receipt_checker(self):
        old_source, replacement = b'authentic checker', b'fabricated checker'
        digest = hashlib.sha256(old_source).hexdigest()
        previous = {'public_sources': {auth.PREFIX: digest}}
        registered = {'public_sources': {auth.PREFIX: digest}}
        auth._check_prior_checker(registered, previous, old_source)
        forged = {'public_sources': {auth.PREFIX: hashlib.sha256(replacement).hexdigest()}}
        with self.assertRaises(ValueError):
            auth._check_prior_checker(forged, previous, replacement)


if __name__ == '__main__':
    unittest.main()
