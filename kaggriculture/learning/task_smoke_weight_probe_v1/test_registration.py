"""Synthetic Git registration; no real source artifacts or game data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from learning.panel_eval_v3 import launch as frozen_launch
from learning.panel_eval_v3.test_launch import synthetic_runtime_repository
from . import run


def encoded(value):
    return (json.dumps(value, sort_keys=True) + '\n').encode()


def put(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


class SyntheticRegistrationTests(unittest.TestCase):
    def test_direct_child_accepts_frozen_bytes_and_rejects_changes(self):
        # The caller's owned scratch encloses this TemporaryDirectory; Git is
        # fully stopped and this directory closes before the outer owner closes.
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            project, repo = base/'project', base/'repository'
            launcher_raw = Path(frozen_launch.__file__).read_bytes()
            needle = ("_REFERENCE_COMMIT = '" + frozen_launch._REFERENCE_COMMIT + "'").encode()
            self.assertEqual(launcher_raw.count(needle), 1)
            def launcher(ref):
                return launcher_raw.replace(needle, ("_REFERENCE_COMMIT = '" + ref + "'").encode())
            reference, smoke_source, ref_freeze, supplemental = synthetic_runtime_repository(
                project, repo, launcher=launcher)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE)
            git('config', 'core.longpaths', 'true')
            def commit(message):
                git('add', '-A')
                git('-c', 'user.name=Synthetic', '-c', 'user.email=synthetic', 'commit', '--quiet', '-m', message)
                return git('rev-parse', 'HEAD').decode().strip()

            smoke = project/'artifacts'/run.SMOKE_STAGE
            smoke_public = repo/'review'/'evidence'/run.SMOKE_STAGE
            smoke_config = dict(run_kind='development_smoke', selected_game_key=run.SMOKE_KEY)
            freeze = dict(ref_freeze, parent_commit=smoke_source, supplemental_runtime_sha256=supplemental)
            for name, value in {'registration.json': smoke_config, 'freeze.json': freeze,
                                'root-decision.json': {}, 'pre-review.json': {},
                                'safety-registration.json': {}}.items():
                raw = encoded(value)
                put(smoke/name, raw)
                put(smoke_public/name, raw)
            smoke_registration = commit('synthetic smoke registration')
            cert_raw = encoded({'synthetic_certificate': True})
            put(repo/run.CERTIFICATE, cert_raw)
            cert_commit = commit('synthetic certificate')
            self.assertIn(run.CERTIFICATE, git('ls-tree', '-r', '--name-only', cert_commit).decode())
            lookup = subprocess.run(['git', '-C', str(repo), 'show', cert_commit+':'+run.CERTIFICATE],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(lookup.returncode, 0, lookup.stderr.decode(errors='replace'))
            self.assertEqual(lookup.stdout, cert_raw)
            sources = {}
            for rel in run.RUNTIME:
                raw = run._project_path(rel).read_bytes()
                sources[rel] = hashlib.sha256(raw).hexdigest()
                put(project/rel, raw)
                put(repo/'kaggriculture'/rel, raw)
            source_commit = commit('synthetic probe source')
            reg = dict(schema='same-observation-weight-probe-v1', status='approved_for_one_offline_probe',
                source_commit=source_commit, smoke_source_commit=smoke_source,
                smoke_registration_commit=smoke_registration, smoke_certificate_commit=cert_commit,
                smoke_game=run.SMOKE_KEY, side=0, sampled_turns=list(run.SAMPLE_TURNS),
                intervention='negate_task_w2_and_b2_in_memory', input_limits=run.INPUT_LIMITS,
                max_controller_calls=run.MAX_CONTROLLER_CALLS, new_games=0, official_transitions=0,
                no_economic_evaluation=True, conclusion_rules=run.CONCLUSION_RULES, supervision=run.SUPERVISION,
                retry_allowed=False, training=False, official_counterfactual_execution=False,
                runtime_source_sha256=sources, smoke_certificate_sha256=hashlib.sha256(cert_raw).hexdigest())
            put(repo/run.REGISTRATION, encoded(reg))
            registration = commit('synthetic direct child registration')
            scratch = base/'scratch'
            scratch.mkdir()
            pycache = str(scratch/'cache')
            flags = sys.flags
            class SyntheticFlags:
                isolated, dont_write_bytecode, optimize = 1, 1, 0
                def __getattr__(self, name):
                    return getattr(flags, name)
            real_samefile = os.path.samefile
            def synthetic_samefile(left, right):
                if str(right) in (str(project/'.venv'), str(project/'.venv'/'Scripts'/'python.exe')):
                    return True
                if str(left) == str(sys.prefix) and str(right) == str(sys.base_prefix):
                    return False
                return real_samefile(left, right)
            argv = ['python', '-B', '-X', 'pycache_prefix='+pycache, '-I', 'synthetic-entry']
            with patch.multiple(run, ROOT=project, PUBLIC=repo, SOURCE_COMMIT=smoke_source,
                                SOURCE_REGISTRATION=smoke_registration, CERTIFICATE_COMMIT=cert_commit), \
                 patch.object(sys, 'orig_argv', argv), patch.object(sys, 'flags', SyntheticFlags()), \
                 patch.object(sys, 'pycache_prefix', pycache), patch.object(os.path, 'samefile', synthetic_samefile):
                accepted, _, config, _ = run._startup(source_commit, registration, scratch)
                self.assertEqual(accepted, reg)
                self.assertEqual(config, smoke_config)
                source_path = project/run.RUNTIME[0]
                before = source_path.read_bytes()
                source_path.write_bytes(before+b'changed')
                with self.assertRaises(ValueError):
                    run._startup(source_commit, registration, scratch)
                source_path.write_bytes(before)
                with self.assertRaises(ValueError):
                    run._startup(source_commit, source_commit, scratch)
                for key, invalid in (('max_controller_calls', 97), ('new_games', 1),
                                     ('official_transitions', 1), ('no_economic_evaluation', False),
                                     ('conclusion_rules', {}), ('supervision', {}),
                                     ('input_limits', dict(run.INPUT_LIMITS, replay_compressed_bytes=float(run.INPUT_LIMITS['replay_compressed_bytes']))),
                                     ('supervision', dict(run.SUPERVISION, wall_seconds=float(run.SUPERVISION['wall_seconds'])))):
                    put(repo/run.REGISTRATION, encoded(dict(reg, **{key: invalid})))
                    git('add', run.REGISTRATION)
                    git('-c', 'user.name=Synthetic', '-c', 'user.email=synthetic',
                        'commit', '--quiet', '--amend', '--no-edit')
                    invalid_registration = git('rev-parse', 'HEAD').decode().strip()
                    with self.subTest(field=key), self.assertRaises(ValueError):
                        run._startup(source_commit, invalid_registration, scratch)
                put(repo/run.REGISTRATION, encoded(reg))
                git('add', run.REGISTRATION)
                git('-c', 'user.name=Synthetic', '-c', 'user.email=synthetic',
                    'commit', '--quiet', '--amend', '--no-edit')
                # A subsequent clean Git history that modifies an old protected
                # source must be refused even if the new S and R are consistent.
                protected = repo/'kaggriculture'/'learning'/'synthetic_frozen'/'runtime.py'
                protected.write_bytes(protected.read_bytes()+b'changed')
                altered_source = commit('alter protected source')
                altered_reg = dict(reg, source_commit=altered_source)
                put(repo/run.REGISTRATION, encoded(altered_reg))
                altered_registration = commit('synthetic altered registration')
                with self.assertRaises(ValueError):
                    run._startup(altered_source, altered_registration, scratch)


if __name__ == '__main__':
    unittest.main()
