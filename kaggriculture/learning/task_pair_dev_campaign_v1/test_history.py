"""Purely synthetic identity and historical-role regressions; no file IO."""
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from . import history


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def payload(value):
    return json.dumps(value, separators=(',', ':')).encode()


def synthetic_checker(extra=''):
    return ("SEED_ALIASES=(('S02',1),('S03',2))\n"
            "OPPONENTS=('round2',)\n"
            "ARMS=('merged','unweighted','teacher_task')\n"
            "def checked_rows(panel,status):\n"
            " return {(a,o,s,p):{'row':{'seed_alias':a,'opponent':o,'side':s,"
            "'policy':p,'seed':1001 if a=='S02' else 1002}} "
            "for a in ('S02','S03') for o in OPPONENTS for s in (0,1) for p in ARMS}\n"
            + extra).encode()


class HistoricalRoles(unittest.TestCase):
    def call(self, source, check=lambda: True, digest=None):
        return history.historical_roles(source=source, source_sha256=digest or sha(source),
            source_path=Path(__file__).resolve(), panel_path='/fake/panel',
            status_path='/fake/status.json', check=check)

    def test_both_consumed_roles_and_two_opponents(self):
        self.assertEqual(self.call(synthetic_checker()), {'S02': 1001, 'S03': 1002})

    def test_rejects_wrong_checker_and_registration(self):
        source = synthetic_checker()
        with self.assertRaises(ValueError):
            self.call(source, digest='0' * 64)
        with self.assertRaises(ValueError):
            self.call(source, check=lambda: False)

    def test_rejects_missing_and_inconsistent_historical_rows(self):
        missing = synthetic_checker("old=checked_rows\n"
            "def checked_rows(panel,status):\n"
            " x=old(panel,status); x.pop(next(iter(x))); return x\n")
        inconsistent = synthetic_checker("old=checked_rows\n"
            "def checked_rows(panel,status):\n"
            " x=old(panel,status); x[next(iter(x))]['row']['seed']=999; return x\n")
        for source in (missing, inconsistent):
            with self.subTest(case=source[-40:]), self.assertRaises(ValueError):
                self.call(source)


class PriorBindings(unittest.TestCase):
    def setUp(self):
        self.identities = dict(baseline='1'*64, control='2'*64, auxiliary='3'*64)
        self.crop_digest, self.market_digest = '4'*64, '5'*64
        self.round2_digest, self.worker_digest = '6'*64, '7'*64
        self.supervisor = payload(dict(schema='task-pair-wsl-supervisor-private-result-v1',
            technical_execution_passed=True, model_files_reloaded=True,
            weights_distinct=True, owned_session_clean=True, new_games=False,
            private_model_sha256=self.identities))
        self.export = payload(dict(checkpoint_sha256=self.identities['baseline']))
        self.seed_document = payload(dict(seeds=[1000, 1001, 1002, 1003]))
        self.model_paths = {a: 'artifacts/fake/' + a + '.npz' for a in self.identities}
        self.cfg = dict(engine_commit=history.ENGINE_COMMIT, engine_sha256=history.ENGINE_SHA,
            seed_document_sha256=sha(self.seed_document),
            crop_checkpoint='artifacts/fake/crop.npz', crop_checkpoint_sha256=self.crop_digest,
            source_sha256={'market.py': self.market_digest, 'round2.py': self.round2_digest})
        self.freeze = {'runtime_files': {'learning/task_anchor_once_v1/games.py': self.worker_digest}}
        self.hashes = {**{self.model_paths[a]: self.identities[a] for a in self.identities},
            self.cfg['crop_checkpoint']: self.crop_digest,
            'artifacts/fake-panel/sources/market.py': self.market_digest,
            'artifacts/fake-panel/sources/round2.py': self.round2_digest,
            'learning/task_anchor_once_v1/games.py': self.worker_digest}

    def call(self, **changes):
        args = dict(seeds={'S02': 1001, 'S03': 1002}, supervisor_raw=self.supervisor,
            supervisor_sha256=sha(self.supervisor), selected_export_raw=self.export,
            selected_export_sha256=sha(self.export), historical_registration=self.cfg,
            historical_freeze=self.freeze, historical_seed_raw=self.seed_document,
            game_root='/fake', historical_panel_relative='artifacts/fake-panel',
            model_paths=self.model_paths, check=lambda: True)
        args.update(changes)
        with patch.object(history, '_safe_relative_file', side_effect=lambda root, rel, *a, **kw: rel), \
             patch.object(history, '_file_digest', side_effect=lambda rel, limit: self.hashes[rel]):
            return history.bind_prior(**args)

    def test_binds_prior_byte_identities_and_both_roles(self):
        manifest = self.call()
        self.assertEqual(manifest.panel.seeds, (1001, 1002))
        self.assertEqual(manifest.opponent('market')[1], self.market_digest)
        self.assertNotIn('1001', repr(manifest))

    def test_rejects_wrong_prior_commitment_and_duplicate_json_keys(self):
        with self.assertRaises(ValueError):
            self.call(supervisor_sha256='0'*64)
        raw = b'{"schema":1,"schema":2}'
        with self.assertRaises(ValueError):
            self.call(supervisor_raw=raw, supervisor_sha256=sha(raw))

    def test_rejects_swapped_consumed_seed_roles(self):
        with self.assertRaises(ValueError):
            self.call(seeds={'S02': 1002, 'S03': 1001})
        with self.assertRaises(ValueError):
            self.call(historical_seed_raw=payload(dict(seeds=[1000, 1002, 1001, 1003])))

    def test_rejects_different_model_and_selected_export(self):
        self.hashes[self.model_paths['control']] = '8'*64
        with self.assertRaises(ValueError):
            self.call()
        self.hashes[self.model_paths['control']] = self.identities['control']
        raw = payload(dict(checkpoint_sha256='9'*64))
        with self.assertRaises(ValueError):
            self.call(selected_export_raw=raw, selected_export_sha256=sha(raw))

    def test_rejects_changed_crop_opponent_worker_and_engine(self):
        for path in (self.cfg['crop_checkpoint'], 'artifacts/fake-panel/sources/market.py',
                     'learning/task_anchor_once_v1/games.py'):
            prior = self.hashes[path]
            self.hashes[path] = '8'*64
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.call()
            self.hashes[path] = prior
        cfg = dict(self.cfg, engine_sha256='0'*64)
        with self.assertRaises(ValueError):
            self.call(historical_registration=cfg)

    def test_invalid_or_changing_registration_never_passes(self):
        with self.assertRaises(ValueError):
            self.call(check=lambda: False)
        values = iter([True, False])
        with self.assertRaises(ValueError):
            self.call(check=lambda: next(values))


if __name__ == '__main__':
    unittest.main()
