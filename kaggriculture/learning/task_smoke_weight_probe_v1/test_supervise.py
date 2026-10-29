"""Synthetic supervisor tests; no real probe data or game is opened."""
from contextlib import contextmanager
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import run, supervise as s


class ContractTests(unittest.TestCase):
    def test_registered_resource_contract_matches_worker(self):
        self.assertEqual(s.SUPERVISION, run.SUPERVISION)
        self.assertEqual(s.SUPERVISION['max_job_commit_bytes'], 4*1024**3)
        self.assertEqual(s.SUPERVISION['min_available_bytes'], 4*1024**3)
        self.assertEqual(s.SUPERVISION['wall_seconds'], 600)
        self.assertFalse(s._same_json(dict(s.SUPERVISION, prelaunch_memory_samples=True), s.SUPERVISION))
        self.assertFalse(s._same_json(dict(s.SUPERVISION, wall_seconds=600.0), s.SUPERVISION))

    def test_memory_sample_fails_closed(self):
        import psutil
        for value in (0, s.SUPERVISION['min_available_bytes']-1, None, True):
            with self.subTest(value=value), patch.object(psutil, 'virtual_memory', return_value=type('M', (), {'available': value})()):
                with self.assertRaises(ValueError):
                    s._memory_ready()
        with patch.object(psutil, 'virtual_memory', return_value=type('M', (), {'available': s.SUPERVISION['min_available_bytes']})()):
            s._memory_ready()

    def test_shared_deadline_does_not_extend_after_clock_rollback(self):
        from datetime import datetime as dt
        now = dt(2026, 1, 1, tzinfo=timezone.utc)
        with patch.object(s.time, 'monotonic', side_effect=[100.0, 101.0, 99.0]), \
             patch.object(s, 'datetime', wraps=dt) as clock:
            clock.now.side_effect = [now, now-timedelta(days=1)]
            deadline = s.SharedDeadline()
            self.assertEqual(deadline.remaining(), 599.0)
            with self.assertRaises(ValueError):
                deadline.remaining()

    def test_child_cannot_pass_without_a_verified_job_limit(self):
        class API:
            limit_verified = False
        result=dict(passed=True,child_reaped=True,job_members_zero=True,handles_closed=True,
                    cleanup_known=True,exit_code=0)
        clock=type('Clock',(),{'remaining':lambda self:10, 'utc_end':datetime.now(timezone.utc)})()
        helper=type('Helper',(),{'default_root':lambda self:Path(tempfile.gettempdir())})()
        with tempfile.TemporaryDirectory() as temp, patch.object(s,'_memory_ready'), \
             patch.object(s,'_limited_job',return_value=API()):
            with self.assertRaises(ValueError):
                s._child({'_supervise':lambda *a,**kw:result},helper,Path(temp),clock,[],Path(temp)/'cache')


class SyntheticGitTests(unittest.TestCase):
    def test_source_whitelist_and_direct_registration_child(self):
        # The outer owned scratch encloses this nested TemporaryDirectory.
        # It closes all Git processes and removes read-only objects first.
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); project = base/'project'; repo = base/'repo'
            repo.mkdir(); project.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE).decode().strip()
            def put(path, raw):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            def commit(message):
                git('add', '-A')
                git('-c', 'user.name=Synthetic', '-c', 'user.email=synthetic', 'commit', '--quiet', '-m', message)
                return git('rev-parse', 'HEAD')
            git('init', '--quiet')
            launcher_rel = 'learning/panel_eval_v3/launch.py'
            launcher = (s.ROOT/launcher_rel).read_bytes()
            put(project/launcher_rel, launcher); put(repo/'kaggriculture'/launcher_rel, launcher)
            put(repo/'review/PROGRESS.md', b'synthetic before\n')
            parent = commit('synthetic base')
            for rel in s.SOURCE_FILES:
                raw = (s.ROOT/rel).read_bytes()
                put(project/rel, raw); put(repo/'kaggriculture'/rel, raw)
            put(repo/'review/PROGRESS.md', b'synthetic after\n')
            put(repo/'review/evidence/task-smoke-weight-probe-v1-source-20261030/validation.json', b'{}\n')
            source = commit('synthetic source')
            registration_path = repo/s.REGISTRATION
            put(registration_path, json.dumps({'source_commit':source,'supervision':s.SUPERVISION}).encode())
            registration = commit('synthetic registration')
            with patch.multiple(s, ROOT=project, PUBLIC=repo, BASE_COMMIT=parent, SMOKE_SOURCE=parent):
                _, digest = s._source_check(source, registration, s.SharedDeadline())
                self.assertEqual(len(digest), 64)
                with self.assertRaises(ValueError):
                    s._source_check(parent, registration, s.SharedDeadline())
                put(repo/'synthetic-extra.txt', b'extra')
                registration_extra = commit('unexpected registered file')
                with self.assertRaises(ValueError):
                    s._source_check(registration, registration_extra, s.SharedDeadline())
                git('checkout', '--quiet', '--detach', source)
                put(repo/'synthetic-source-extra.txt', b'extra')
                git('add', '-A')
                git('-c','user.name=Synthetic','-c','user.email=synthetic','commit','--quiet','--amend','--no-edit')
                altered_source = git('rev-parse', 'HEAD')
                put(registration_path, json.dumps({'source_commit':altered_source,'supervision':s.SUPERVISION}).encode())
                altered_registration = commit('synthetic altered registration')
                with self.assertRaises(ValueError):
                    s._source_check(altered_source, altered_registration, s.SharedDeadline())


class SyntheticAttemptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'project'
        (self.root/'artifacts').mkdir(parents=True)
        self.source, self.registration, self.digest = 'a'*40, 'b'*40, 'c'*64

    def helper(self, *, fail_cleanup=False):
        root = self.root
        class Helper:
            def default_root(self): return root
            def read_marker(self, path, root): return {'run_id':'d'*32}
            @contextmanager
            def scratch_directory(self, purpose):
                if fail_cleanup:
                    path=root/'synthetic-retained'; path.mkdir()
                    try:
                        yield path
                    finally:
                        raise OSError('synthetic cleanup failure')
                    return
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    yield path
        return Helper()

    def invoke(self, mode, *, failure=False, fail_cleanup=False):
        helper = self.helper(fail_cleanup=fail_cleanup)
        def child(_launcher, _helper, scratch, _clock, command, _prefix):
            if failure: raise RuntimeError('synthetic failure')
            if '-c' in command:
                s._write(scratch/'ready.json', dict(source_commit=self.source, registration_commit=self.registration,
                                                   entry_sha256=self.digest, scope=s._READY_SCOPE))
            else:
                target=self.root/'artifacts'/s.STAGE_NAME
                target.mkdir()
                s._write(target/'result.json', dict(passed=True,source_commit=self.source,
                                                    registration_commit=self.registration))
            return {'passed':True}
        with patch.object(s, 'ROOT', self.root), patch.object(s, '_startup'), \
             patch.object(s, '_source_check', return_value=({},self.digest)), patch.object(s, '_load_launcher', return_value={}), \
             patch.object(s, '_load_helper', return_value=helper), patch.object(s, '_child', side_effect=child):
            return s.supervise(mode,self.source,self.registration,'synthetic-helper')

    def test_verify_then_run_requires_ready_and_is_single_use(self):
        self.assertTrue(self.invoke('verify')['passed'])
        before=self.registration;self.registration='d'*40
        with self.assertRaises(ValueError): self.invoke('run')
        self.registration=before
        self.assertTrue(self.invoke('run')['passed'])
        with self.assertRaises(ValueError): self.invoke('run')

    def test_invalid_or_missing_verification_refuses_before_any_child_or_attempt(self):
        good=dict(passed=True,source_commit=self.source,registration_commit=self.registration,
                  cleanup_known=True,retry_allowed=False)
        cases=(('missing_directory',None),('missing_result',{}),('failed',dict(good,passed=False)),
               ('wrong_source',dict(good,source_commit='d'*40)),
               ('wrong_registration',dict(good,registration_commit='e'*40)),
               ('numeric_passed',dict(good,passed=1)),('numeric_cleanup',dict(good,cleanup_known=1)),
               ('numeric_retry',dict(good,retry_allowed=0)))
        for label, value in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)/'project';artifacts=root/'artifacts';artifacts.mkdir(parents=True)
                verify=artifacts/(s.STAGE_NAME+'-supervision-verify')
                if label != 'missing_directory':
                    verify.mkdir()
                    if label != 'missing_result': s._write(verify/'result.local.json',value)
                with patch.object(s,'ROOT',root), patch.object(s,'_startup'), \
                     patch.object(s,'_source_check') as source, patch.object(s,'_load_launcher') as launcher, \
                     patch.object(s,'_child') as child:
                    with self.assertRaises((ValueError,FileNotFoundError)):
                        s.supervise('run',self.source,self.registration,'synthetic-helper')
                    source.assert_not_called();launcher.assert_not_called();child.assert_not_called()
                self.assertFalse((artifacts/(s.STAGE_NAME+'-supervision-attempt')).exists())

    def test_linked_verification_directory_is_rejected_before_any_child(self):
        verify=self.root/'artifacts'/(s.STAGE_NAME+'-supervision-verify');verify.mkdir()
        s._write(verify/'result.local.json',dict(passed=True,source_commit=self.source,
            registration_commit=self.registration,cleanup_known=True,retry_allowed=False))
        original=Path.lstat
        def synthetic_lstat(path):
            info=original(path)
            if path == verify:
                return SimpleNamespace(st_mode=info.st_mode,
                    st_file_attributes=getattr(info,'st_file_attributes',0)|0x400)
            return info
        with patch.object(s,'ROOT',self.root), patch.object(s,'_startup'), \
             patch.object(Path,'lstat',synthetic_lstat), patch.object(s,'_source_check') as source, \
             patch.object(s,'_child') as child:
            with self.assertRaises(ValueError):
                s.supervise('run',self.source,self.registration,'synthetic-helper')
            source.assert_not_called();child.assert_not_called()
        self.assertFalse((self.root/'artifacts'/(s.STAGE_NAME+'-supervision-attempt')).exists())

    def test_pre_marker_rejection_can_be_corrected_without_consuming_verification(self):
        artifacts=self.root/'artifacts'
        with patch.object(s,'ROOT',self.root), patch.object(s,'_startup'), \
             patch.object(s,'_source_check',side_effect=ValueError('synthetic source rejection')):
            with self.assertRaises(ValueError):
                s.supervise('verify',self.source,self.registration,'synthetic-helper')
        self.assertFalse((artifacts/(s.STAGE_NAME+'-supervision-verify')).exists())
        with patch.object(s,'ROOT',self.root), patch.object(s,'_startup'), patch.object(s,'_source_check') as source:
            with self.assertRaises(ValueError): s.supervise('verify','invalid',self.registration,'synthetic-helper')
            source.assert_not_called()
        self.assertFalse((artifacts/(s.STAGE_NAME+'-supervision-verify')).exists())
        self.assertTrue(self.invoke('verify')['passed'])

    def test_deadline_after_cleanup_blocks_success_and_retry(self):
        values=iter((1.0,0.0))
        clock=type('Clock',(),{'remaining':lambda self:next(values)})()
        with patch.object(s,'SharedDeadline',return_value=clock):
            self.assertFalse(self.invoke('verify')['passed'])
        with self.assertRaises((ValueError,FileNotFoundError)): self.invoke('run')

    def test_failed_verify_blocks_run_even_if_cleanup_succeeds(self):
        self.assertFalse(self.invoke('verify',failure=True)['passed'])
        with self.assertRaises((ValueError,FileNotFoundError)): self.invoke('run')

    def test_failed_or_unknown_run_never_allows_retry_or_reports_path(self):
        self.assertTrue(self.invoke('verify')['passed'])
        result=self.invoke('run',failure=True,fail_cleanup=True)
        self.assertFalse(result['passed'])
        self.assertFalse(result['cleanup_known'])
        self.assertNotIn('scratch_path',json.dumps(result))
        with self.assertRaises(ValueError): self.invoke('run')


@unittest.skipUnless(os.name == 'nt', 'Windows Job required')
class WindowsJobTests(unittest.TestCase):
    def test_job_commit_limit_is_configured_and_queried(self):
        launcher = runpy.run_path(str(s.ROOT/'learning/panel_eval_v3/launch.py'))
        api = s._limited_job(launcher)
        # NUL stdio is provided by the exact previously validated base method.
        self.assertIs(type(api).create_suspended, launcher['WindowsJob'].create_suspended)
        job = api.create_job()
        try:
            api.configure(job)
            self.assertTrue(api.limit_verified)
            value = launcher['_ExtendedLimit']()
            self.assertTrue(api.k.QueryInformationJobObject(job,9,ctypes.byref(value),ctypes.sizeof(value),None))
            self.assertEqual(value.JobMemoryLimit,s.SUPERVISION['max_job_commit_bytes'])
            self.assertEqual(value.BasicLimitInformation.LimitFlags & (s._KILL_ON_CLOSE|s._LIMIT_JOB_MEMORY),
                             s._KILL_ON_CLOSE|s._LIMIT_JOB_MEMORY)
        finally:
            api.close(job)

    @unittest.skipUnless(os.environ.get('PANEL_ARTIFACTS_HELPER'), 'Standalone helper must be supplied')
    def test_salted_helper_and_owned_verification_child(self):
        launcher = runpy.run_path(str(s.ROOT/'learning/panel_eval_v3/launch.py'))
        helper_path = Path(os.environ['PANEL_ARTIFACTS_HELPER'])
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp)/'stage';stage.mkdir()
            document={'version':1,'seeds':[11,22,33,44],'salt':'ab'*32}
            raw=json.dumps(document).encode();(stage/'private-seeds.local.json').write_bytes(raw)
            helper_raw=helper_path.read_bytes()
            commitment=hashlib.sha256(b'panel-scratch-helper-v1\x00'+bytes.fromhex(document['salt'])+
                                      hashlib.sha256(helper_raw).digest()).hexdigest()
            cfg=dict(seed_document_sha256=hashlib.sha256(raw).hexdigest(),scratch_helper_commitment=commitment)
            helper,_=launcher['_helper'](helper_path,stage,cfg,dict(cfg))
            self.assertEqual(helper.sweep_stale(),{'scanned':0,'removed':[],'bytes':0})
            with helper.scratch_directory('synthetic-weight-supervisor') as scratch:
                entry=scratch/'synthetic-entry.py'
                entry.write_text("from pathlib import Path\n"
                    "def _startup(source,registration,scratch): Path(scratch,'first').write_text('ok')\n"
                    "def _namespace_and_runtime(scratch):\n"
                    " assert Path(scratch,'first').read_text() == 'ok'\n"
                    " Path(scratch,'second').write_text('ok')\n",encoding='utf-8')
                source,registration,digest='a'*40,'b'*40,'c'*64
                prefix=scratch/'verify-cache'
                command=[sys.executable,'-B','-X','pycache_prefix='+str(prefix),'-I','-c',s._VERIFY_CODE,
                         str(entry),source,registration,str(scratch),digest]
                with patch.object(s,'_memory_ready'):
                    result=s._child(launcher,helper,scratch,s.SharedDeadline(),command,prefix)
                self.assertTrue(result['passed'])
                self.assertEqual(json.loads((scratch/'ready.json').read_text()),
                    dict(source_commit=source,registration_commit=registration,entry_sha256=digest,scope=s._READY_SCOPE))
                self.assertTrue((scratch/'second').exists())
            self.assertFalse(scratch.exists())


if __name__ == '__main__':
    unittest.main()
