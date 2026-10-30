"""Private-binding and child-command regressions with synthetic memory data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from . import bootstrap
from .manifest import build_manifest


def payload(value):
    return json.dumps(value, separators=(',', ':')).encode()


class Bootstrap(unittest.TestCase):
    def manifest(self):
        return build_manifest(seeds={'S02': 1001, 'S03': 1002},
            models={role: {'path': 'artifacts/example/' + role + '.npz', 'sha256': digit * 64}
                    for role, digit in (('baseline', '1'), ('control', '2'), ('auxiliary', '3'))},
            crop={'path': 'artifacts/example/crop.npz', 'sha256': '4'*64},
            opponents={'market': {'path': 'artifacts/example/market.py', 'sha256': '5'*64},
                       'round2': {'path': 'artifacts/example/round2.py', 'sha256': '6'*64}},
            worker_sha256='7'*64)

    def test_manifest_document_preserves_all_prebound_roles(self):
        value = self.manifest()
        doc = bootstrap._manifest_document(value, '8'*40, '9'*64)
        self.assertEqual(doc['seeds'], {'S02': 1001, 'S03': 1002})
        self.assertEqual(doc['models']['auxiliary']['sha256'], '3'*64)
        self.assertEqual(doc['opponents']['market']['sha256'], '5'*64)
        self.assertEqual(doc['prebinding_sha256'], '9'*64)

    def test_private_binding_must_match_earlier_precommitment(self):
        binding = dict(schema='task-pair-dev-prebinding-v1',
            prior_training_result_commit='5a66216f99a8e467818978c96e707f6045834709',
            historical_result_commit='3a3e3172b0403a171a38f6e78e37d07f387be2cb',
            seed_roles=['S02', 'S03'], seed_scope='previously_consumed_development')
        data = payload(binding)
        status = dict(current_pair_dev_diagnostic={'prebinding_record_sha256': hashlib.sha256(data).hexdigest()},
            current_pair_training={'wsl_contracts': {'public_result_commit': binding['prior_training_result_commit'],
                'attempt_status': 'completed_exit_observed', 'operator_observed_wrapper_exit_code': 0}})
        def reader(path, maximum=bootstrap.MAX_PRIVATE):
            return data if path.name == bootstrap.PREBIND else payload(status)
        with patch.object(bootstrap, '_read', side_effect=reader):
            self.assertEqual(bootstrap._private_status()[3], binding)
            status['current_pair_dev_diagnostic']['prebinding_record_sha256'] = '0'*64
            with self.assertRaises(ValueError):
                bootstrap._private_status()

    def test_child_command_contains_alias_but_no_seed_or_model_digest(self):
        command = bootstrap._command('synthetic-root', '8'*40, 'game', 'dev-S02-market-0-baseline')
        self.assertEqual(command[-2:], ['--game', 'dev-S02-market-0-baseline'])
        text = ' '.join(command)
        self.assertNotIn('1001', text)
        self.assertNotIn('1'*64, text)
        self.assertIn('data!=result.stdout', text)
        projection = bootstrap._command('synthetic-root', '8'*40, 'project')
        self.assertEqual(projection[-1], '--project')

    def test_identity_requires_the_authenticated_in_memory_snapshot(self):
        with patch.dict(bootstrap.__dict__, {'_AUTH_IDENTITY_BYTES': None}):
            with self.assertRaises(ValueError):
                bootstrap._identity()
        with patch.dict(bootstrap.__dict__, {'_AUTH_IDENTITY_BYTES': b'VALUE = 42\n'}), \
             patch.object(bootstrap, '_read', side_effect=AssertionError('unexpected identity reread')):
            self.assertEqual(bootstrap._identity().VALUE, 42)

    def test_fresh_isolated_process_has_dependency_finder(self):
        source_root = str(Path(bootstrap.__file__).resolve().parents[2])
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "from learning.task_pair_dev_campaign_v1 import bootstrap; "
                "assert callable(bootstrap.importlib.util.find_spec)")
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code, source_root],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))

    def test_thread_environment_is_fixed_before_numpy_and_inherited(self):
        source_root = str(Path(bootstrap.__file__).resolve().parents[2])
        code = ("import os,subprocess,sys; sys.path.insert(0,sys.argv[1]); "
                "from learning.task_pair_dev_campaign_v1 import bootstrap; "
                "bootstrap._configure_thread_environment(); "
                "assert os.environ['OPENBLAS_NUM_THREADS']==os.environ['OMP_NUM_THREADS']=='1'; "
                "check=\"import os; assert os.environ['OPENBLAS_NUM_THREADS']==os.environ['OMP_NUM_THREADS']=='1'\"; "
                "subprocess.run([sys.executable,'-I','-B','-c',check],check=True); "
                "sys.modules['numpy']=object(); "
                "exec(\"try:\\n bootstrap._configure_thread_environment()\\nexcept ValueError:\\n pass\\nelse:\\n raise AssertionError('numpy was already imported')\")")
        env = dict(__import__('os').environ, OPENBLAS_NUM_THREADS='7', OMP_NUM_THREADS='9')
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code, source_root],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertEqual((env['OPENBLAS_NUM_THREADS'], env['OMP_NUM_THREADS']), ('7', '9'))

    def test_private_module_shadow_is_rejected_before_import(self):
        root = Path('synthetic-game')
        shadow = root / 'learning' / 'task_pair_dev_games_v1.py'
        with patch.object(bootstrap.os.path, 'lexists', side_effect=lambda path: path == shadow):
            with self.assertRaisesRegex(ValueError, 'module shadow'):
                bootstrap._check_new_packages(root, {})

    def test_invalid_limits_stop_both_entries_before_private_status(self):
        identity = Mock()
        identity.verify_public.return_value = ({'limits': {}}, {}, {})
        campaign = Mock()
        campaign.limits_snapshot.side_effect = ValueError('invalid limits')
        with patch.object(bootstrap, '_absolute', return_value=Path('synthetic-game')), \
             patch.object(bootstrap, '_identity', return_value=identity), \
             patch.object(bootstrap, '_runtime', return_value={'campaign': campaign}), \
             patch.object(bootstrap, '_private_status') as private:
            for entry in (bootstrap.prepare, bootstrap.execute, bootstrap.project_evidence):
                with self.subTest(entry=entry.__name__), self.assertRaisesRegex(ValueError, 'invalid limits'):
                    entry(game_root='synthetic-game', registration_commit='8'*40)
            private.assert_not_called()

    def test_recheck_rejects_changed_manifest_or_attempt_state(self):
        manifest = payload({'synthetic': 1})
        binding = {'synthetic': 2}
        dev = dict(state='running', registration_commit='8'*40, stage_name=bootstrap.STAGE_NAME,
                   manifest_sha256=hashlib.sha256(manifest).hexdigest())
        class Identity:
            @staticmethod
            def _git(_repo, operation, *args):
                return ('8'*40 + '\n').encode() if operation == 'rev-parse' else b''
        with patch.object(bootstrap, '_private_status', return_value=(None, {}, dev, binding)), \
             patch.object(bootstrap, '_read', return_value=manifest):
            args = dict(identity=Identity(), game_root=None, registration_commit='8'*40,
                        manifest_raw=manifest, binding=binding, required_state='running')
            self.assertTrue(bootstrap._recheck(**args))
            dev['state'] = 'finished_no_promotion'
            self.assertFalse(bootstrap._recheck(**args))
            dev['state'] = 'running'
            dev['manifest_sha256'] = '0'*64
            self.assertFalse(bootstrap._recheck(**args))

    def test_checks_every_frozen_asset_and_refuses_changed_model(self):
        manifest = self.manifest()
        expected = dict(manifest.panel.models)
        expected[manifest.panel.crop[0]] = manifest.panel.crop[1]
        for role in ('market', 'round2'):
            relative, digest = manifest.opponent(role)
            expected[relative] = digest
        expected['learning/task_anchor_once_v1/games.py'] = manifest.panel.worker_sha
        class History:
            MODEL_LIMIT = METADATA_LIMIT = 100
            @staticmethod
            def _safe_relative_file(_root, relative, *_args, **_kwargs):
                return relative
            @staticmethod
            def _file_digest(relative, _limit):
                return expected[relative]
        bootstrap._verify_assets(manifest, None, History())
        expected[manifest.panel.models[2][0]] = '0'*64
        with self.assertRaises(ValueError):
            bootstrap._verify_assets(manifest, None, History())


if __name__ == '__main__':
    unittest.main()
