from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import campaign as c
from . import experiment as e
from .evaluation import plans


class CampaignTests(unittest.TestCase):
    def test_storage_reserves_job_before_launch(self):
        cfg={'storage':dict(game_write_bytes=100,process_log_bytes=10,total_bytes=5000000,free_floor_bytes=1000)}
        reserve=100+10+4*c.MIB
        with patch.object(c,'owned_size',return_value=100),patch.object(c.shutil,'disk_usage',return_value=SimpleNamespace(free=1000+reserve)):
            self.assertEqual(c.storage_ready(Path('.'),cfg),100)
        with patch.object(c,'owned_size',return_value=5000000-reserve+1):
            with self.assertRaises(RuntimeError): c.storage_ready(Path('.'),cfg)
        with patch.object(c,'owned_size',return_value=100),patch.object(c.shutil,'disk_usage',return_value=SimpleNamespace(free=999+reserve)):
            with self.assertRaises(RuntimeError): c.storage_ready(Path('.'),cfg)

    def test_memory_samples_and_stops_on_low_memory(self):
        cfg={'host_preflight':dict(samples=3,min_available_bytes=10,interval_seconds=10)}
        values=iter([10,11,12]); sleeps=[]
        with patch.object(c,'deadline_check'):
            self.assertTrue(c.memory_ready(cfg,available=lambda:next(values),sleep=sleeps.append))
        self.assertEqual(sleeps,[10,10])
        values=iter([10,9]); sleeps=[]
        with patch.object(c,'deadline_check'):
            self.assertFalse(c.memory_ready(cfg,available=lambda:next(values),sleep=sleeps.append))
        self.assertEqual(sleeps,[10])

    def test_preflight_identity_drift_stops_before_marker(self):
        cfg={'host_preflight':dict(samples=3,min_available_bytes=4*1024**3),
             'official_job_supervision':dict(min_available_bytes=2*1024**3)}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(c,'OUT',Path(tmp)),patch.object(c,'ready',return_value=cfg),patch.object(c,'verify_commit'), \
                 patch.object(c,'trusted_out',side_effect=lambda path:Path(path)), \
                 patch.object(c,'memory_ready',return_value=True),patch.object(c,'recheck',side_effect=ValueError('changed')), \
                 patch.object(c,'marker') as marker,patch.object(c,'supervise') as launch:
                with self.assertRaises(ValueError): c.run(Path(tmp))
                marker.assert_not_called(); launch.assert_not_called()

    def test_teacher_task_acceptance_has_no_task_checkpoint(self):
        digest='a'*64
        plan=dict(game='holdout-S01-teacher-0-teacher_task',seed_alias='S01',seed=1111111111,
            opponent='teacher',side=0,policy='teacher_task',split='holdout',parity_kind=None,
            bot_sha256=digest,rival_sha256=digest,task_checkpoint_sha256=None)
        row={**plan,**dict(passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
            bank=3,opponent_bank=2,initial_bank=1,opponent_initial_bank=1,net_profit=2,
            opponent_net_profit=1,margin=1,score=1,replay_gzip_sha256=digest,trace_sha256=digest)}
        cfg=dict(crop_checkpoint='crop.npz',crop_checkpoint_sha256=digest,source_sha256={'teacher.py':digest})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            def read(path): return row if str(path).endswith('.row.json') else {}
            with patch.object(c,'validate_plan'),patch.object(c,'validate_receipts'),patch.object(c,'verify_economy'), \
                 patch.object(c,'read',side_effect=read),patch.object(c,'sha',return_value=digest), \
                 patch.object(c,'task_identity',return_value=(None,None)):
                saved,audit=c.accept_job(out,plan,cfg)
                self.assertEqual(saved['policy'],'teacher_task')
                self.assertIsNone(audit['task_checkpoint_sha256'])
                with self.assertRaises(ValueError): c.accept_job(out,{**plan,'policy':'merged'},cfg)

    def test_later_preflight_identity_drift_does_not_start_next_game(self):
        cfg={'host_preflight':dict(samples=3,min_available_bytes=4*1024**3),
             'official_job_supervision':dict(min_available_bytes=2*1024**3)}
        queue=plans([1111111111,2222222222,3333333333,4000000001])
        def mark(out,state): state.update(created=True,complete=True)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with ExitStack() as stack:
                stack.enter_context(patch.object(c,'OUT',out))
                stack.enter_context(patch.object(c,'trusted_out',side_effect=lambda path:Path(path)))
                stack.enter_context(patch.object(c,'ready',return_value=cfg))
                stack.enter_context(patch.object(c,'verify_commit'))
                stack.enter_context(patch.object(c,'memory_ready',return_value=True))
                stack.enter_context(patch.object(c,'storage_ready'))
                stack.enter_context(patch.object(c,'load_seeds',return_value=[1111111111,2222222222,3333333333,4000000001]))
                stack.enter_context(patch.object(c,'plans',return_value=queue))
                stack.enter_context(patch.object(c,'marker',side_effect=mark))
                stack.enter_context(patch.object(c,'identity',return_value={'freeze.json':'x'}))
                stack.enter_context(patch.object(c,'recheck',side_effect=[None,None,None,ValueError('changed')]))
                stack.enter_context(patch.object(c,'make_wrapper',return_value=out/'wrapper'))
                stack.enter_context(patch.object(c,'task_identity',return_value=(None,None)))
                stack.enter_context(patch.object(c,'sha',return_value='x'*64))
                stack.enter_context(patch.object(c,'validate_plan'))
                written=stack.enter_context(patch.object(c,'write_new'))
                stack.enter_context(patch.object(c,'WindowsAPI'))
                launch=stack.enter_context(patch.object(c,'supervise',return_value={'passed':True}))
                stack.enter_context(patch.object(c,'job_ok',return_value=True))
                stack.enter_context(patch.object(c,'check_identity'))
                stack.enter_context(patch.object(c,'accept_job',return_value=({'game':'one'},{'game':'one'})))
                with self.assertRaises(c.CampaignFailure): c.run(out)
                self.assertEqual(launch.call_count,1)
                self.assertTrue(any(str(call.args[0]).endswith('games-failure.local.json') for call in written.call_args_list))

    def test_complete_panel_reports_python_and_native_output_separately(self):
        cfg={'host_preflight':dict(samples=3,min_available_bytes=4*1024**3),
             'official_job_supervision':dict(min_available_bytes=2*1024**3)}
        seeds=[1111111111,2222222222,3333333333,4000000001]; queue=plans(seeds)
        def mark(out,state): state.update(created=True,complete=True)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with ExitStack() as stack:
                stack.enter_context(patch.object(c,'OUT',out))
                stack.enter_context(patch.object(c,'trusted_out',side_effect=lambda path:Path(path)))
                stack.enter_context(patch.object(c,'ready',return_value=cfg))
                stack.enter_context(patch.object(c,'verify_commit'))
                stack.enter_context(patch.object(c,'memory_ready',return_value=True))
                stack.enter_context(patch.object(c,'storage_ready'))
                stack.enter_context(patch.object(c,'load_seeds',return_value=seeds))
                stack.enter_context(patch.object(c,'plans',return_value=queue))
                stack.enter_context(patch.object(c,'marker',side_effect=mark))
                stack.enter_context(patch.object(c,'identity',return_value={'freeze.json':'x'}))
                stack.enter_context(patch.object(c,'recheck'))
                stack.enter_context(patch.object(c,'make_wrapper',return_value=out/'wrapper'))
                stack.enter_context(patch.object(c,'task_identity',return_value=(None,None)))
                stack.enter_context(patch.object(c,'sha',return_value='x'*64))
                stack.enter_context(patch.object(c,'validate_plan'))
                written=stack.enter_context(patch.object(c,'write_new'))
                stack.enter_context(patch.object(c,'WindowsAPI'))
                launch=stack.enter_context(patch.object(c,'supervise',return_value={'passed':True}))
                stack.enter_context(patch.object(c,'job_ok',return_value=True))
                stack.enter_context(patch.object(c,'check_identity'))
                stack.enter_context(patch.object(c,'accept_job',return_value=({'game':'one'},{'game':'one'})))
                stack.enter_context(patch.object(c,'verify_ledger'))
                stack.enter_context(patch.object(c,'deadline_check'))
                result=c.run(out)
            self.assertEqual(launch.call_count,72)
            self.assertIs(result['python_stdout_stderr_discarded'],True)
            self.assertIs(result['native_output_handles_independently_verified'],False)
            self.assertNotIn('worker_stdout_stderr_discarded',result)
            final=[call.args[1] for call in written.call_args_list if str(call.args[0]).endswith('games-result.json')]
            self.assertEqual(final,[result])

    def test_panel_rejects_overstated_output_receipt(self):
        result=dict(passed=True,complete_games=72,live_official_transitions=72*719,primary_replay_transitions=72*719,
            python_stdout_stderr_discarded=True,native_output_handles_independently_verified=False,
            game_ledger_sha256='sha',audit_ledger_sha256='sha',identities={})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            def read(path): return result if Path(path).name=='games-result.json' else []
            with patch.object(c,'ready'),patch.object(c,'verify_commit'),patch.object(c,'read',side_effect=read), \
                 patch.object(c,'trusted_out',side_effect=lambda path:Path(path)), \
                 patch.object(c,'sha',return_value='sha'),patch.object(c,'verify_ledger',return_value=['accepted']):
                self.assertEqual(c.validate_panel(out,{}),['accepted'])
                result['native_output_handles_independently_verified']=True
                with self.assertRaises(ValueError): c.validate_panel(out,{})
                result['native_output_handles_independently_verified']=False
                result['worker_stdout_stderr_discarded']=True
                with self.assertRaises(ValueError): c.validate_panel(out,{})

    def test_marker_rejects_stage_junction_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; artifacts=root/'artifacts'; artifacts.mkdir(parents=True)
            outside=Path(tmp)/'outside'; outside.mkdir(); sentinel=outside/'sentinel'; sentinel.write_bytes(b'keep')
            stage=artifacts/e.STAGE_NAME
            if os.name=='nt':
                try:
                    subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(stage),str(outside)],
                                   check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                except (OSError,subprocess.CalledProcessError) as error:
                    self.skipTest('Directory junction unavailable: '+type(error).__name__)
                if getattr(stage,'is_junction',None) is not None: self.assertTrue(stage.is_junction())
            else:
                try: stage.symlink_to(outside,target_is_directory=True)
                except (OSError,NotImplementedError) as error: self.skipTest('Directory link unavailable: '+type(error).__name__)
            try:
                with patch.object(e,'ROOT',root),patch.object(e,'OUT',stage):
                    state=dict(created=False,complete=False)
                    with self.assertRaises(ValueError): c.marker(stage,state)
                    with self.assertRaises(ValueError): c.run(stage)
                    with self.assertRaises(ValueError): c.validate_panel(stage,{})
                self.assertEqual(state,dict(created=False,complete=False))
                self.assertEqual(sentinel.read_bytes(),b'keep')
                self.assertFalse((outside/'games-started.json').exists())
            finally:
                stage.rmdir() if os.name=='nt' else stage.unlink()

    def test_marker_creates_only_in_legal_stage_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; stage=root/'artifacts'/e.STAGE_NAME; stage.mkdir(parents=True)
            (stage/'freeze.json').write_bytes(b'freeze')
            with patch.object(e,'ROOT',root),patch.object(e,'OUT',stage):
                state=dict(created=False,complete=False); c.marker(stage,state)
                marker=stage/'games-started.json'; original=marker.read_bytes()
                self.assertEqual(json.loads(original)['freeze_sha256'],e.sha(stage/'freeze.json'))
                self.assertEqual(state,dict(created=True,complete=True))
                repeat=dict(created=False,complete=False)
                with self.assertRaises(FileExistsError): c.marker(stage,repeat)
                self.assertEqual(repeat,dict(created=False,complete=False))
                self.assertEqual(marker.read_bytes(),original)


if __name__=='__main__': unittest.main()
