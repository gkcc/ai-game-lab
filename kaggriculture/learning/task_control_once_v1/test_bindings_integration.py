"""In-memory integration of historical role and asset binding.

The real historical checker's ``checked_rows`` and ``check_selected`` code
objects run against synthetic, cryptographically linked receipts. Public
completion screening and filesystem operations are virtualized; this is not
an audit of the real historical receipts or proof of registration identity.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace
import unittest
from unittest import mock

from learning.task_pair_dev_campaign_v1 import history, manifest as manifest_module
from learning.task_pair_dev_games_v1.panel import ENGINE_COMMIT, ENGINE_SHA, GAME_LIMITS
from learning.task_prefix_diagnostic_v2 import runner as historical
from . import bindings


def encoded(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


class MemoryPath:
    def __init__(self, path, files):
        self.path, self.files = path, files

    def stat(self):
        return SimpleNamespace(st_size=len(self.files[self.path]))


class Fixture:
    """Link 72 virtual ledger rows to genuine checker-compatible receipts."""
    def __init__(self, bad_s01_row=False):
        self.root = Path.cwd() / 'virtual-game'
        self.repo = Path.cwd() / 'virtual-repo'
        self.panel = 'panel'
        self.files = {}
        self.model_files = {}
        self.teacher_bytes = b'synthetic teacher source'
        self.teacher_sha = digest(self.teacher_bytes)
        self.worker_bytes = b'synthetic worker source'
        self.worker_sha = digest(self.worker_bytes)
        self.seed_raw = encoded({'seeds': [1000, 1001, 1002, 1003]})
        self.model_data = {role: ('synthetic model ' + role).encode()
                           for role in ('baseline', 'control', 'auxiliary', 'crop')}
        self.model_ids = {role: digest(data) for role, data in self.model_data.items()}
        self.source_bytes = {'teacher': self.teacher_bytes,
                             'market': b'synthetic market source',
                             'round2': b'synthetic round2 source'}
        self.source_ids = {role: digest(data) for role, data in self.source_bytes.items()}
        self.cfg = dict(engine_commit=ENGINE_COMMIT, engine_sha256=ENGINE_SHA,
                        seed_document_sha256=digest(self.seed_raw),
                        merged_checkpoint_sha256='1' * 64,
                        unweighted_checkpoint_sha256='2' * 64,
                        crop_checkpoint='artifacts/models/crop.npz',
                        crop_checkpoint_sha256=self.model_ids['crop'],
                        source_sha256={role + '.py': value for role, value in self.source_ids.items()},
                        game_limits=dict(GAME_LIMITS),
                        storage={'game_write_bytes': 128 * 1024**2})
        self.freeze = {'runtime_files': {'learning/task_anchor_once_v1/games.py': self.worker_sha}}
        self.supervisor = encoded(dict(schema='task-pair-wsl-supervisor-private-result-v1',
            technical_execution_passed=True, model_files_reloaded=True, weights_distinct=True,
            owned_session_clean=True, new_games=False,
            private_model_sha256={role: self.model_ids[role]
                                  for role in ('baseline', 'control', 'auxiliary')}))
        self.export = encoded({'checkpoint_sha256': self.model_ids['baseline']})
        self.source = ("SEED_ALIASES=(('S02',1),('S03',2))\n"
                       "OPPONENTS=('round2',)\n"
                       "ARMS=('merged','unweighted','teacher_task')\n").encode()
        self.reg = {'public_sources': {bindings.PREFIX: digest(self.source),
                    'synthetic-export.json': digest(self.export),
                    'kaggriculture/' + bindings.TEACHER_REL: self.teacher_sha}}
        self.prior = {'prior_supervisor_sha256': digest(self.supervisor)}
        self.bootstrap = SimpleNamespace(
            OLD_PANEL='artifacts/historical', SUPERVISOR='artifacts/models/supervisor.json',
            MODEL_DIR='artifacts/models', EXPORT='synthetic-export.json', _read=self.private_read)
        self.identity = SimpleNamespace(_read=self.public_read)
        self.modules = {'history': history, 'manifest': manifest_module}
        for role, data in self.model_data.items():
            self.model_files['artifacts/models/' + role + '.npz'] = data
        self.model_files.update({'artifacts/historical/sources/' + role + '.py': data
                                 for role, data in self.source_bytes.items()})
        self.model_files[bindings.TEACHER_REL] = self.teacher_bytes
        self.model_files['learning/task_anchor_once_v1/games.py'] = self.worker_bytes
        self._historical_files(bad_s01_row)

    def put(self, name, value):
        raw = value if type(value) is bytes else encoded(value)
        self.files[name] = raw
        return digest(raw)

    def _historical_files(self, bad_s01_row):
        cfg_raw, freeze_raw = encoded(self.cfg), encoded(self.freeze)
        self.put('registration', cfg_raw)
        self.put('panel/registration.json', cfg_raw)
        self.put('panel/freeze.json', freeze_raw)
        self.put('panel/private-seeds.local.json', self.seed_raw)
        ledger, audits = [], []
        for alias_idx, alias in enumerate(('S01', 'S02', 'S03', 'S04')):
            for opponent in ('teacher', 'market', 'round2'):
                self.put('panel/sources/' + opponent + '.py', self.source_bytes[opponent])
                for side in (0, 1):
                    for policy in ('merged', 'unweighted', 'teacher_task'):
                        key = f'holdout-{alias}-{opponent}-{side}-{policy}'
                        seed = 1000 + alias_idx
                        row_seed = 9999 if (bad_s01_row and alias == 'S01' and opponent == 'teacher'
                                            and side == 0 and policy == 'merged') else seed
                        task = None if policy == 'teacher_task' else self.cfg[policy + '_checkpoint_sha256']
                        wrapper = ('synthetic agent ' + key).encode()
                        self.put('panel/agents/' + key + '.py', wrapper)
                        replay_id, trace_id = digest(('replay' + key).encode()), digest(('trace' + key).encode())
                        sem1, sem2 = digest(('semantic1' + key).encode()), digest(('semantic2' + key).encode())
                        plan = dict(game=key, seed_alias=alias, seed=row_seed, opponent=opponent,
                                    side=side, policy=policy)
                        row = dict(plan, states=720, action_turns=719, status=['DONE', 'DONE'], agent_errors=0,
                                   task_checkpoint_sha256=task, crop_checkpoint_sha256=self.model_ids['crop'],
                                   engine_sha256=ENGINE_SHA, rival_sha256=self.source_ids[opponent],
                                   bot_sha256=digest(wrapper), replay_gzip_sha256=replay_id, trace_sha256=trace_id,
                                   semantic_replay_sha256=sem1, semantic_replay_sha256_v2=sem2, rewards=[10, 20])
                        counts = dict(task_forwards=0 if task is None else 3,
                                      task_rows=0 if task is None else 3, crop_forwards=1, crop_rows=1,
                                      gate_evaluations=0, controller_calls=719, teacher_proposals=0)
                        official = dict(passed=True, official_transitions_reexecuted=719, engine_sha256=ENGINE_SHA,
                            both_public_and_private_states_match_each_step=True,
                            inventory_order_checked_each_step=True, semantic_sha256=sem1,
                            semantic_sha256_v2=sem2, final_rewards=row['rewards'])
                        audit = dict(passed=True, game=key, errors=0, actor_actions_reproduced=719,
                                     rival_actions_reproduced=719, trace_turns_reproduced=719,
                                     replay_sha256=replay_id, trace_sha256=trace_id, official=official,
                                     primary_model_counts=counts, live_model_counts=counts)
                        storage = dict(passed=True, game=key, error_kind=None,
                                       output=dict(limit_bytes=128 * 1024**2, used_bytes=12,
                                                   remaining_bytes=128 * 1024**2 - 12, refused=0))
                        paths = 'panel/job-results/' + key + '.'
                        row_sha = self.put(paths + 'row.json', row)
                        audit_sha = self.put(paths + 'audit.json', audit)
                        storage_sha = self.put(paths + 'storage.json', storage)
                        complete_sha = self.put(paths + 'complete.json',
                            dict(passed=True, game=key, row_sha256=row_sha, certificate_sha256=audit_sha))
                        input_sha = self.put('panel/job-inputs/' + key + '.json', plan)
                        job = dict(passed=True, assigned=True, resumed=True, launcher_reaped=True,
                                   job_members_zero=True, exit_code=0, error_kind=None,
                                   cleanup_error_kind=None, stop_reason=None)
                        supervisor_sha = self.put('panel/job-supervision/' + key + '.json',
                            dict(game=key, freeze_sha256=digest(freeze_raw), job_input_sha256=input_sha,
                                 retry_allowed=False, supervision=job))
                        evidence = dict(game=key, official_transitions=719, replay_sha256=replay_id,
                                        trace_sha256=trace_id, task_checkpoint_sha256=task,
                                        receipt_sha256=dict(row=row_sha, audit=audit_sha,
                                                            complete=complete_sha, storage=storage_sha),
                                        job_supervision=job)
                        acceptance_sha = self.put('panel/job-acceptance/' + key + '.json',
                            dict(passed=True, game=key, retry_allowed=False, freeze_sha256=digest(freeze_raw),
                                 job_input_sha256=input_sha, supervision_sha256=supervisor_sha, evidence=evidence))
                        ledger.append({k: row[k] for k in ('game', 'seed_alias', 'opponent', 'side', 'policy')})
                        audits.append(dict(game=key, acceptance_receipt_sha256=acceptance_sha,
                                           replay_sha256=replay_id, trace_sha256=trace_id,
                                           supervision_receipt_sha256=supervisor_sha, job_supervision=job))
                        self.put('panel/games/' + key + '/replay.json.gz', b'virtual replay')
                        self.put('panel/traces/' + key + '.jsonl.gz', b'virtual trace')
        self.put('ledger', ledger)
        audit_ledger_sha = self.put('panel/game-audit-ledger.json', audits)
        self.put('panel/games-result.json', dict(passed=True, complete_games=72,
                    live_official_transitions=72 * 719, primary_replay_transitions=72 * 719,
                    audit_ledger_sha256=audit_ledger_sha))
        self.ledger_sha, self.cfg_sha, self.freeze_sha = (digest(self.files[k]) for k in
                                                          ('ledger', 'registration', 'panel/freeze.json'))

    def read_small(self, path):
        return self.files[path.path]

    def safe_file(self, _root, *parts):
        name = 'panel/' + '/'.join(parts)
        if name not in self.files:
            raise ValueError('Synthetic input missing')
        return MemoryPath(name, self.files)

    def checker_module(self, name):
        fixture = self
        class Checker(ModuleType):
            def __getattribute__(self, key):
                if key == 'checked_rows':
                    scope = object.__getattribute__(self, '__dict__')
                    scope.update(dict(LEDGER=MemoryPath('ledger', fixture.files),
                        REGISTRATION=MemoryPath('registration', fixture.files), LEDGER_SHA256=fixture.ledger_sha,
                        REGISTRATION_SHA256=fixture.cfg_sha, FREEZE_SHA256=fixture.freeze_sha,
                        read_small=fixture.read_small, safe_file=fixture.safe_file, json=json,
                        verify_completion=lambda _panel, _status: None,
                        sha=historical.sha, lf_sha=historical.lf_sha, require=historical.require,
                        equal=historical.equal, check_counts=historical.check_counts))
                    scope['load_small'] = FunctionType(historical.load_small.__code__, scope,
                                                       argdefs=historical.load_small.__defaults__)
                    scope['check_selected'] = FunctionType(historical.check_selected.__code__, scope)
                    return FunctionType(historical.checked_rows.__code__, scope)
                return super().__getattribute__(key)
        return Checker(name)

    def private_read(self, path, maximum):
        values = {self.root / self.bootstrap.OLD_PANEL / 'private-seeds.local.json': self.seed_raw,
                  self.root / self.bootstrap.SUPERVISOR: self.supervisor,
                  self.repo / self.bootstrap.EXPORT: self.export}
        raw = values[path]
        if len(raw) > maximum:
            raise ValueError('Synthetic input too large')
        return raw

    def public_read(self, path):
        values = {self.repo / bindings.PREFIX: self.source,
                  self.repo / 'kaggriculture' / bindings.TEACHER_REL: self.teacher_bytes}
        return values[path]

    def asset_path(self, _root, relative, _suffix, **_kwargs):
        if relative not in self.model_files:
            raise ValueError('Synthetic asset missing')
        return relative

    def asset_digest(self, relative, maximum):
        raw = self.model_files[relative]
        if not raw or len(raw) > maximum:
            raise ValueError('Synthetic asset size')
        return digest(raw)

    def build(self, check=lambda: True):
        with mock.patch.object(bindings, 'ModuleType', self.checker_module), \
             mock.patch.object(history, 'ModuleType', self.checker_module), \
             mock.patch.object(history, '_safe_relative_file', self.asset_path), \
             mock.patch.object(history, '_file_digest', self.asset_digest), \
             mock.patch.object(bindings, 'TEACHER_SHA', self.teacher_sha):
            return bindings.build(modules=self.modules, bootstrap=self.bootstrap, identity=self.identity,
                reg=self.reg, old_reg=self.cfg, freeze=self.freeze, game_root=self.root, repo=self.repo,
                status_path=self.root / 'virtual-status.json', prior_binding=self.prior, check=check)


class BuildIntegrationTests(unittest.TestCase):
    def test_build_binds_s01_teacher_and_frozen_assets(self):
        fixture = Fixture()
        manifest, previous, seed, teacher = fixture.build()
        self.assertEqual(seed, 1000)
        self.assertEqual(manifest.panel.seeds, (1000, 1001))
        self.assertEqual(manifest.opponent('market'), ('artifacts/historical/sources/teacher.py', fixture.teacher_sha))
        self.assertEqual(manifest.panel.models, previous.panel.models)
        self.assertEqual(manifest.panel.models[1][1], fixture.model_ids['control'])
        self.assertEqual(manifest.panel.crop[1], fixture.model_ids['crop'])
        self.assertEqual(manifest.panel.worker_sha, fixture.worker_sha)
        self.assertEqual(teacher, 'artifacts/historical/sources/teacher.py')

    def test_wrong_s01_receipt_seed_fails_genuine_historical_row_check(self):
        with self.assertRaises(ValueError):
            Fixture(bad_s01_row=True).build()

    def test_control_crop_teacher_and_worker_byte_mismatches_stop(self):
        for relative in ('artifacts/models/control.npz', 'artifacts/models/crop.npz',
                         'artifacts/historical/sources/teacher.py', bindings.TEACHER_REL,
                         'learning/task_anchor_once_v1/games.py'):
            with self.subTest(kind=relative):
                fixture = Fixture()
                fixture.model_files[relative] = b'different synthetic bytes'
                with self.assertRaises(ValueError):
                    fixture.build()

    def test_missing_or_changed_evidence_and_external_check_stop(self):
        fixture = Fixture()
        fixture.files.pop('panel/private-seeds.local.json')
        with self.assertRaises(ValueError):
            fixture.build()
        fixture = Fixture()
        fixture.prior['prior_supervisor_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            fixture.build()
        with self.assertRaises(ValueError):
            Fixture().build(check=lambda: False)
        calls = iter((True, True, False))
        with self.assertRaises(ValueError):
            Fixture().build(check=lambda: next(calls))


if __name__ == '__main__':
    unittest.main()
