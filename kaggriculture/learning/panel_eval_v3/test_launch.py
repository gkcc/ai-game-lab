"""Synthetic Git and Windows Job regressions; never launch a real panel."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv

from . import launch as l, experiment as e


def encoded(value):
    return (json.dumps(value,indent=2)+'\n').encode()


def synthetic_runtime_repository(project, repo, *, launcher=None, bootstrap=None):
    """Create reference, source and later registration commits in a real Git repo."""
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE)

    def put(path, raw):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)

    repo.mkdir(parents=True)
    project.mkdir(parents=True)
    git('init','--quiet')
    runtime = {'learning/synthetic_frozen/runtime.py':b'synthetic historical runtime\n',
               'learning/synthetic_frozen/new.py':b'synthetic historical source\n'}
    for rel, raw in runtime.items():
        put(project/rel,raw)
        put(repo/'kaggriculture'/rel,raw)
    historical = b'{"synthetic": true}\n'
    put(repo/'review'/'evidence'/'synthetic-history'/'registration.json',historical)
    runtime_rel, new_rel = tuple(runtime)
    reference = dict(runtime_files={runtime_rel:l._sha(runtime[runtime_rel])},
        public_runtime_sha256={runtime_rel:l._sha(runtime[runtime_rel])},
        new_source_sha256={new_rel:l._sha(runtime[new_rel])},
        historical_evidence_sha256={'synthetic-history/registration.json':l._sha(historical)})
    put(repo/l._REFERENCE_FREEZE,encoded(reference))
    git('add','kaggriculture','review')
    git('-c','user.name=Test','-c','user.email=synthetic','commit','--quiet','-m','reference')
    reference_commit=git('rev-parse','HEAD').decode().strip()
    launcher = launcher(reference_commit) if callable(launcher) else (launcher or b'synthetic launch.py')
    bootstrap = bootstrap or b'synthetic bootstrap.py'
    identities={}
    for rel in sorted(l._SUPPLEMENTAL_SOURCES):
        raw=(launcher if rel == 'learning/panel_eval_v3/launch.py' else
             bootstrap if rel == 'learning/panel_eval_v3/bootstrap.py' else
             ('synthetic '+rel+'\n').encode())
        put(project/rel,raw)
        put(repo/'kaggriculture'/rel,raw)
        digest=l._sha(raw)
        identities[rel]={'local_sha256':digest,'public_sha256':digest}
    git('add','kaggriculture')
    git('-c','user.name=Test','-c','user.email=synthetic','commit','--quiet','-m','source')
    parent=git('rev-parse','HEAD').decode().strip()
    return reference_commit,parent,reference,identities


class SourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.project=self.base/'project'; self.repo=self.base/'repo'
        self.reference_commit,self.parent,reference,supplemental=synthetic_runtime_repository(self.project,self.repo)
        reference_patch=patch.object(l,'_REFERENCE_COMMIT',self.reference_commit)
        reference_patch.start(); self.addCleanup(reference_patch.stop)
        self.stage=self.project/'artifacts'/'synthetic-panel'; self.stage.mkdir(parents=True)
        self.cfg={'baseline_head':self.parent,'deadline_utc':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
                  'outer_supervision':dict(l.OUTER_SUPERVISION),'environment':{'interpreter_sha256':'a'*64},
                  'scratch_helper_commitment':'e'*64}
        self.cfg.update(e.run_contract(e.FORMAL))
        decision={'scratch_helper_commitment':'e'*64,'run_kind':e.FORMAL}
        review=dict(decision,passed=True,review_basis='independent_subagent',reviewer='/root/synthetic',
                    open_findings=[],checks=['source_identity'])
        self.freeze=dict(reference,parent_commit=self.parent,run_kind=e.FORMAL,
                         supplemental_runtime_sha256=supplemental)
        files={'root-decision.json':encoded(decision),'registration.json':encoded(self.cfg),
               'freeze.json':encoded(self.freeze),
               'pre-review.json':encoded(review),'safety-registration.json':encoded({})}
        dest=self.repo/'review'/'evidence'/self.stage.name; dest.mkdir(parents=True)
        for name,raw in files.items():
            (self.stage/name).write_bytes(raw); (dest/name).write_bytes(raw)
        self.git('add','review')
        self.git('-c','user.name=Test','-c','user.email=synthetic','commit','--quiet','-m','registration')
        self.commit=self.git('rev-parse','HEAD').decode().strip()
        (self.stage/'pre-execution-commit.json').write_bytes(encoded(dict(commit=self.commit,
            registration_sha256=hashlib.sha256(files['registration.json']).hexdigest(),
            freeze_sha256=hashlib.sha256(files['freeze.json']).hexdigest())))

    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.repo),*args],stderr=subprocess.PIPE)

    def verify(self):
        return l._verify_sources(self.repo,self.stage,self.commit,self.project,
                                 self.project/'learning'/'panel_eval_v3'/'launch.py')

    def test_clean_direct_child_and_exact_source_bytes_are_accepted(self):
        cfg, decision, identities=self.verify()
        self.assertEqual(cfg,self.cfg)
        self.assertEqual(decision,{'scratch_helper_commitment':'e'*64,'run_kind':e.FORMAL})
        self.assertEqual(set(identities),{'launch.py','bootstrap.py'})
        self.assertEqual(identities['launch.py'],hashlib.sha256(b'synthetic launch.py').hexdigest())

    def test_standalone_launcher_contract_matches_registered_experiment(self):
        self.assertEqual(l._SUPPLEMENTAL_SOURCES,e.SUPPLEMENTAL_FILES)
        self.assertEqual(l._REVIEW_CHECK_CODES,e.REVIEW_CHECK_CODES)

    def test_changed_project_source_public_source_or_commit_is_rejected(self):
        for path in (self.project/'learning'/'panel_eval_v3'/'launch.py',
                     self.repo/'kaggriculture'/'learning'/'panel_eval_v3'/'bootstrap.py'):
            before=path.read_bytes(); path.write_bytes(before+b'changed')
            with self.subTest(path=path.name), self.assertRaises(ValueError): self.verify()
            path.write_bytes(before)
        with self.assertRaises(ValueError):
            l._verify_sources(self.repo,self.stage,self.parent,self.project,self.project/'learning'/'panel_eval_v3'/'launch.py')

    def test_changed_worker_or_smoke_runner_stops_before_launch(self):
        for name in ('smoke_runner.py','worker.py'):
            path=self.project/'learning'/'panel_eval_v3'/name
            before=path.read_bytes()
            try:
                path.write_bytes(before+b'changed')
                with self.subTest(name=name), \
                     patch.object(l,'__file__',str(self.project/'learning'/'panel_eval_v3'/'launch.py')), \
                     patch.object(l,'_startup'), patch.object(l,'_verify_interpreter') as interpreter, \
                     patch.object(l,'_helper') as helper, patch.object(l,'_supervise') as supervise:
                    with self.assertRaises(ValueError):
                        l.launch(self.repo,self.stage,self.commit,self.base/'synthetic-helper')
                    interpreter.assert_not_called()
                    helper.assert_not_called()
                    supervise.assert_not_called()
                    self.assertFalse((self.stage/'replays').exists())
            finally:
                path.write_bytes(before)

    def test_missing_frozen_source_or_manifest_entry_is_rejected(self):
        path=self.project/'learning'/'panel_eval_v3'/'worker.py'
        before=path.read_bytes()
        try:
            path.unlink()
            with self.assertRaises((ValueError,OSError)): self.verify()
        finally:
            path.write_bytes(before)
        bad=dict(self.freeze)
        bad['supplemental_runtime_sha256']=dict(self.freeze['supplemental_runtime_sha256'])
        del bad['supplemental_runtime_sha256']['learning/panel_eval_v3/smoke_runner.py']
        raw=encoded(bad)
        (self.stage/'freeze.json').write_bytes(raw)
        (self.repo/'review'/'evidence'/self.stage.name/'freeze.json').write_bytes(raw)
        self.git('add','review/evidence/'+self.stage.name+'/freeze.json')
        self.git('-c','user.name=Test','-c','user.email=synthetic','commit','--quiet','--amend','--no-edit')
        self.commit=self.git('rev-parse','HEAD').decode().strip()
        binding=json.loads((self.stage/'pre-execution-commit.json').read_text())
        binding.update(commit=self.commit,freeze_sha256=l._sha(raw))
        (self.stage/'pre-execution-commit.json').write_bytes(encoded(binding))
        with self.assertRaisesRegex(ValueError,'Supplemental source set is incomplete'): self.verify()

    def test_modified_stage_evidence_and_commit_binding_are_rejected(self):
        path=self.stage/'registration.json'; before=path.read_bytes(); path.write_bytes(before+b' ')
        with self.assertRaises(ValueError): self.verify()
        path.write_bytes(before)
        path=self.stage/'pre-execution-commit.json'
        receipt=json.loads(path.read_text()); receipt['commit']='0'*40
        path.write_bytes(encoded(receipt))
        with self.assertRaises(ValueError): self.verify()

    def test_clean_commit_with_other_changes_is_rejected(self):
        (self.repo/'other.txt').write_text('synthetic')
        self.git('add','other.txt')
        self.git('-c','user.name=Test','-c','user.email=synthetic','commit','--quiet','-m','other')
        later=self.git('rev-parse','HEAD').decode().strip()
        with self.assertRaises(ValueError):
            l._verify_sources(self.repo,self.stage,later,self.project,self.project/'learning'/'panel_eval_v3'/'launch.py')

    def test_duplicate_metadata_and_expired_deadline_are_rejected(self):
        with self.assertRaises(ValueError): l._json(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError): l._deadline('2000-01-01T00:00:00+00:00')
        with self.assertRaises(ValueError): l._deadline('2099-01-01T00:00:00')

    def test_invalid_cli_arguments_do_not_echo_private_markers(self):
        command = [sys.executable,'-B','-I',str(Path(l.__file__).resolve()),
                   '--repo','synthetic','--stage','synthetic','--prereg','synthetic','--helper','synthetic',
                   '--unexpected','SYNTHETIC_PRIVATE_MARKER']
        result = subprocess.run(command, capture_output=True, check=False)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual((result.stdout,result.stderr),(b'',b''))

    def test_review_and_decision_must_match_helper_commitment(self):
        good={'scratch_helper_commitment':'a'*64}
        passed=dict(good,passed=True,review_basis='independent_subagent',reviewer='/root/synthetic',
                    open_findings=[],checks=['source_identity'])
        l._verify_helper_commitment(good,good,passed)
        invalid_reviews=({}, {**passed,'scratch_helper_commitment':'b'*64}, {**passed,'passed':False},
                         {**passed,'review_basis':'webchat'}, {**passed,'reviewer':'/root/'},
                         {**passed,'open_findings':['unresolved']}, {**passed,'checks':[]},
                         {**passed,'checks':['Synthetic test']}, {**passed,'reviewer':'/root/synthetic reviewer'})
        pairs=[({},passed),({'scratch_helper_commitment':'b'*64},passed)]+[(good,review) for review in invalid_reviews]
        for decision,review in pairs:
            with self.subTest(decision=decision,review=review), self.assertRaises(ValueError):
                l._verify_helper_commitment(good,decision,review)
        with self.assertRaises(ValueError): l._verify_helper_commitment({'scratch_helper_commitment':'bad'},good,passed)


class SupervisorSyntheticTests(unittest.TestCase):
    class Helper:
        def __init__(self): self.calls=[]
        def protect_children(self,path,children,*,root,complete): self.calls.append(complete)
        def process_identity(self,pid): return ('active','windows:12345')

    class API:
        def __init__(self): self.terminated=False
        def create_job(self): return 'job'
        def configure(self,job): pass
        def create_suspended(self,command,cwd): return ('process','thread',42)
        def assign(self,job,process): pass
        def resume(self,thread): pass
        def wait(self,process,ms): return True
        def active(self,job): return 0
        def exit_code(self,process): return 0
        def terminate_job(self,job): self.terminated=True
        def close(self,handle): pass

    class Clock:
        def __init__(self,values): self.values=iter(values)
        def remaining(self):
            value=next(self.values)
            if isinstance(value,BaseException): raise value
            return value

    def test_reaped_exit_at_deadline_does_not_pass(self):
        helper=self.Helper(); api=self.API(); clock=self.Clock([3,2,0])
        result=l._supervise(api,['synthetic'],'cwd',datetime.now(timezone.utc)+timedelta(seconds=3),
                            helper,'scratch','root',_clock=clock)
        self.assertFalse(result['passed'])
        self.assertEqual(result['stop_reason'],'registered_hard_deadline')
        self.assertTrue(api.terminated)
        self.assertTrue(result['cleanup_known'])

    def test_unreliable_clock_after_spawn_preserves_incomplete_inventory(self):
        helper=self.Helper(); api=self.API(); clock=self.Clock([3,2,ValueError('synthetic clock failure')])
        result=l._supervise(api,['synthetic'],'cwd',datetime.now(timezone.utc)+timedelta(seconds=3),
                            helper,'scratch','root',_clock=clock)
        self.assertFalse(result['passed'])
        self.assertFalse(result['cleanup_known'])
        self.assertEqual(helper.calls,[False,False])

    def test_wall_clock_rollback_cannot_extend_monotonic_deadline(self):
        start=datetime(2026,1,1,tzinfo=timezone.utc)
        utc=iter([start,start-timedelta(days=1)])
        mono=iter([100.0,102.0])
        clock=l._Deadline(start+timedelta(seconds=1),utc_now=lambda:next(utc),monotonic=lambda:next(mono))
        self.assertLess(clock.remaining(),0)
        mono=iter([100.0,99.0])
        clock=l._Deadline(start+timedelta(seconds=1),utc_now=lambda:start,monotonic=lambda:next(mono))
        with self.assertRaises(ValueError): clock.remaining()

    def test_reaped_leader_with_live_descendants_does_not_busy_poll(self):
        class DescendantAPI(self.API):
            def __init__(self):
                super().__init__(); self.states=iter([1,1,0])
            def active(self,job): return next(self.states)
        class Clock:
            def remaining(self): return 5.0
        helper=self.Helper(); api=DescendantAPI()
        with patch.object(l.time,'sleep') as sleep:
            result=l._supervise(api,['synthetic'],'cwd',datetime.now(timezone.utc)+timedelta(seconds=5),
                                helper,'scratch','root',_clock=Clock())
        self.assertTrue(result['passed'])
        self.assertEqual(sleep.call_count,2)
        self.assertTrue(all(0 < call.args[0] <= .1 for call in sleep.call_args_list))

    def test_unknown_spawn_keeps_incomplete_child_inventory(self):
        class FakeHelper:
            def __init__(self): self.calls=[]
            def protect_children(self,path,children,*,root,complete): self.calls.append(complete)
        class FakeAPI:
            def create_job(self): return object()
            def configure(self,job): pass
            def create_suspended(self,command,cwd): raise OSError('synthetic')
            def close(self,handle): pass
        helper=FakeHelper()
        result=l._supervise(FakeAPI(),['synthetic'],'cwd',datetime.now(timezone.utc)+timedelta(seconds=2),
                            helper,'scratch','root')
        self.assertFalse(result['passed'])
        self.assertFalse(result['cleanup_known'])
        self.assertEqual(helper.calls,[False])

    def test_close_failure_keeps_incomplete_child_inventory(self):
        class FakeHelper:
            def __init__(self): self.calls=[]
            def protect_children(self,path,children,*,root,complete): self.calls.append(complete)
        class FakeAPI:
            def create_job(self): return object()
            def configure(self,job): raise ValueError('synthetic')
            def close(self,handle): raise OSError('synthetic')
        helper=FakeHelper()
        result=l._supervise(FakeAPI(),['synthetic'],'cwd',datetime.now(timezone.utc)+timedelta(seconds=2),
                            helper,'scratch','root')
        self.assertFalse(result['cleanup_known'])
        self.assertEqual(helper.calls,[False])

    def test_uncommitted_helper_cannot_execute_side_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); stage=root/'stage'; stage.mkdir()
            private=encoded({'version':1,'seeds':[11,22,33,44],'salt':'ab'*32})
            (stage/'private-seeds.local.json').write_bytes(private)
            sentinel=root/'side-effect'
            source=root/'helper.py'
            source.write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').write_text("unsafe")\n')
            cfg={'seed_document_sha256':hashlib.sha256(private).hexdigest(),'scratch_helper_commitment':'0'*64}
            decision=dict(cfg)
            with self.assertRaises(ValueError): l._helper(source,stage,cfg,decision)
            self.assertFalse(sentinel.exists())
            (stage/'private-seeds.local.json').write_bytes(private+b' ')
            with self.assertRaises(ValueError): l._helper(source,stage,cfg,decision)
            self.assertFalse(sentinel.exists())

    def test_failure_before_scratch_creation_records_unknown_not_retained(self):
        class FailureContext:
            def __enter__(self): raise OSError('synthetic')
            def __exit__(self,*args): pass
        class Helper:
            def default_root(self): return Path(tempfile.gettempdir())
            def scratch_directory(self,purpose): return FailureContext()
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp)/'stage'; stage.mkdir()
            cfg={'deadline_utc':(datetime.now(timezone.utc)+timedelta(seconds=30)).isoformat()}
            cfg.update(e.run_contract(e.FORMAL))
            sources={'launch.py':'a'*64,'bootstrap.py':'b'*64}
            with patch.object(l,'_startup'), patch.object(l,'_verify_sources',return_value=(cfg,{},sources)), \
                 patch.object(l,'_verify_interpreter',return_value=(Path('synthetic'),'c'*64)), \
                 patch.object(l,'_helper',return_value=(Helper(),'d'*64)):
                self.assertFalse(l.launch(Path(temp),stage,'e'*40,Path(temp)/'helper.py'))
            result=json.loads((stage/'replays'/'launcher-result.json').read_text())
            self.assertIsNone(result['scratch_retained'])
            self.assertIsNone(result['scratch_run_id_if_retained'])
            self.assertIsNone(result['scratch_path_if_retained'])

    def test_confirmed_retained_scratch_path_is_only_in_private_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); stage=root/'stage'; stage.mkdir()
            scratch=root/'synthetic-scratch'; scratch.mkdir()
            class FailureContext:
                def __enter__(self): return scratch
                def __exit__(self,*args): raise OSError('synthetic failed cleanup')
            class Helper:
                def default_root(self): return root
                def scratch_directory(self,purpose): return FailureContext()
                def read_marker(self,path,root): return {'run_id':'f'*32}
            cfg={'deadline_utc':(datetime.now(timezone.utc)+timedelta(seconds=30)).isoformat()}
            cfg.update(e.run_contract(e.FORMAL))
            sources={'launch.py':'a'*64,'bootstrap.py':'b'*64}
            with patch.object(l,'_startup'), patch.object(l,'_verify_sources',return_value=(cfg,{},sources)), \
                 patch.object(l,'_verify_interpreter',return_value=(Path('synthetic'),'c'*64)), \
                 patch.object(l,'_helper',return_value=(Helper(),'d'*64)), patch.object(l,'WindowsJob'), \
                 patch.object(l,'_supervise',return_value={'passed':False,'cleanup_known':False,'retry_allowed':False}):
                self.assertFalse(l.launch(root,stage,'e'*40,root/'helper.py'))
            result=json.loads((stage/'replays'/'launcher-result.json').read_text())
            self.assertTrue(result['scratch_retained'])
            self.assertEqual(result['scratch_path_if_retained'],str(scratch))
            self.assertEqual(result['scratch_run_id_if_retained'],'f'*32)
            self.assertTrue(result['private_receipt_never_publish'])


@unittest.skipUnless(os.name=='nt' and os.environ.get('PANEL_ARTIFACTS_HELPER'),
                     'Windows and explicitly supplied standalone scratch helper required')
class WindowsJobTests(unittest.TestCase):
    def setUp(self):
        helper_path=l._absolute(os.environ['PANEL_ARTIFACTS_HELPER'])
        raw=l._read(helper_path,4*l._SMALL)
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp)
            private=encoded({'version':1,'seeds':[11,22,33,44],'salt':'ab'*32})
            (stage/'private-seeds.local.json').write_bytes(private)
            commitment=hashlib.sha256(b'panel-scratch-helper-v1\x00'+bytes.fromhex('ab'*32)+hashlib.sha256(raw).digest()).hexdigest()
            cfg={'seed_document_sha256':hashlib.sha256(private).hexdigest(),'scratch_helper_commitment':commitment}
            self.helper,_=l._helper(helper_path,stage,cfg,dict(cfg))
        self.root=self.helper.default_root()
        self.project=Path(l.__file__).resolve().parents[2]

    def run_in_scratch(self,command,deadline):
        with self.helper.scratch_directory('panel-launch-synthetic-test') as scratch:
            result=l._supervise(l.WindowsJob(),command(scratch),self.project,deadline,self.helper,scratch,self.root)
        self.assertFalse(scratch.exists())
        return result

    def test_existing_inner_supervisor_runs_inside_outer_job(self):
        code=('import sys\nsys.path.insert(0,sys.argv[1])\n'
              'from learning.task_job_empty_exit_v1.sampling import supervise\n'
              'from learning.task_training_job_supervision_v1.job import WindowsAPI\n'
              'limits=dict(max_rss_bytes=10*1024**3,max_commit_bytes=10*1024**3,min_available_bytes=1,'
              'wall_seconds=12,poll_seconds=.1,cleanup_seconds=3)\n'
              'child=[sys.executable,"-B","-I","-c","import time; time.sleep(0.8)"]\n'
              'r=supervise(WindowsAPI(),child,sys.argv[1],limits)\n'
              'sys.exit(0 if r.get("passed") and r.get("job_members_zero") and r.get("launcher_reaped") else 1)')
        def command(scratch):
            return [sys.executable,'-B','-X','pycache_prefix='+str(scratch/'cache'),'-I','-c',code,str(self.project)]
        result=self.run_in_scratch(command,datetime.now(timezone.utc)+timedelta(seconds=30))
        self.assertTrue(result['passed'])
        self.assertTrue(result['job_members_zero'])
        self.assertTrue(result['child_reaped'])
        self.assertTrue(result['cleanup_known'])

    def test_expired_hard_deadline_terminates_only_owned_job_and_cleans(self):
        def command(_):
            return [sys.executable,'-B','-I','-c','import time; time.sleep(20)']
        result=self.run_in_scratch(command,datetime.now(timezone.utc)+timedelta(seconds=.5))
        self.assertFalse(result['passed'])
        self.assertEqual(result['stop_reason'],'registered_hard_deadline')
        self.assertTrue(result['child_reaped'])
        self.assertTrue(result['job_members_zero'])
        self.assertTrue(result['cleanup_known'])

    def test_direct_launcher_end_to_end_uses_committed_sources_and_isolated_child(self):
        """Synthetic bootstrap writes a fake completion only; no game imports."""
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp); project=base/'project'; repo=base/'repo'
            launcher=Path(l.__file__).read_bytes()
            bootstrap=b'''import json, os, sys\nfrom pathlib import Path\nassert sys.orig_argv[1:3] == ["-B", "-X"]\nassert sys.orig_argv[4] == "-I" and sys.flags.isolated == 1\nassert sys.argv[1] == "learning.panel_eval_v3.runner"\nargs=sys.argv\nscratch=Path(args[args.index("--scratch")+1])\nstage=Path(args[args.index("--out")+1])\nassert Path(sys.pycache_prefix).parent == scratch\nassert not Path(sys.pycache_prefix).exists()\nassert os.environ.get("TEMP") != str(scratch) and os.environ.get("TMP") != str(scratch)\nresults=stage/"results"\nresults.mkdir()\n(results/"coordinator-complete.json").write_text(json.dumps({"passed":True,"economic_evaluation_completed":True}))\n'''
            def fixture_launcher(reference_commit):
                previous=("_REFERENCE_COMMIT = '"+l._REFERENCE_COMMIT+"'").encode()
                replacement=("_REFERENCE_COMMIT = '"+reference_commit+"'").encode()
                self.assertEqual(launcher.count(previous),1)
                return launcher.replace(previous,replacement,1)
            _,parent,reference,supplemental=synthetic_runtime_repository(
                project,repo,launcher=fixture_launcher,bootstrap=bootstrap)
            folder=project/'learning'/'panel_eval_v3'
            venv.EnvBuilder(with_pip=False).create(project/'.venv')
            executable=project/'.venv'/'Scripts'/'python.exe'
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE)
            stage=project/'artifacts'/'synthetic-panel'; stage.mkdir(parents=True)
            private=encoded({'version':1,'seeds':[11,22,33,44],'salt':'ab'*32})
            (stage/'private-seeds.local.json').write_bytes(private)
            helper=Path(os.environ['PANEL_ARTIFACTS_HELPER'])
            commitment=hashlib.sha256(b'panel-scratch-helper-v1\x00'+bytes.fromhex('ab'*32)+hashlib.sha256(helper.read_bytes()).digest()).hexdigest()
            common={'seed_document_sha256':hashlib.sha256(private).hexdigest(),'scratch_helper_commitment':commitment,
                    'run_kind':e.FORMAL}
            cfg=dict(common,baseline_head=parent,deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat(),
                     outer_supervision=l.OUTER_SUPERVISION,environment={'interpreter_sha256':l._sha(executable.read_bytes())})
            cfg.update(e.run_contract(e.FORMAL))
            review=dict(common,passed=True,review_basis='independent_subagent',reviewer='/root/synthetic',
                        open_findings=[],checks=['source_identity'])
            files={'root-decision.json':encoded(common),'registration.json':encoded(cfg),
                   'freeze.json':encoded(dict(reference,parent_commit=parent,run_kind=e.FORMAL,
                                              supplemental_runtime_sha256=supplemental)),
                   'pre-review.json':encoded(review),
                   'safety-registration.json':encoded({})}
            evidence=repo/'review'/'evidence'/stage.name; evidence.mkdir(parents=True)
            for name,data in files.items():
                (stage/name).write_bytes(data); (evidence/name).write_bytes(data)
            git('add','review'); git('-c','user.name=Test','-c','user.email=synthetic',
                                     'commit','--quiet','-m','registration')
            commit=git('rev-parse','HEAD').decode().strip()
            (stage/'pre-execution-commit.json').write_bytes(encoded({'commit':commit,
                'registration_sha256':l._sha(files['registration.json']), 'freeze_sha256':l._sha(files['freeze.json'])}))
            command=[str(executable),'-B','-I',str(folder/'launch.py'),
                     '--repo',str(repo),'--stage',str(stage),'--prereg',commit,'--helper',str(helper)]
            first=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,timeout=30)
            self.assertEqual(first.returncode,0)
            self.assertEqual((first.stdout,first.stderr),(b'',b''))
            result=json.loads((stage/'replays'/'launcher-result.json').read_text())
            self.assertTrue(result['passed'])
            self.assertTrue(result['job_members_zero'])
            self.assertFalse(result['scratch_retained'])
            second=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,timeout=30)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual((second.stdout,second.stderr),(b'',b''))


if __name__ == '__main__':
    unittest.main()
