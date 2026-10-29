"""Synthetic development-smoke boundaries; no real match or formal seed."""
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_teacher_merged_eval_v1 import campaign, games, independent_runner
from . import experiment as e, isolation, launch, prepare, runner, smoke_runner, worker
from . import test_prepare as prepare_fixtures


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


class SeedSeparationTests(unittest.TestCase):
    def fixture(self):
        fixture=prepare_fixtures.PreparationTests('test_existing_stage_is_never_adopted_and_no_rng_is_called')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_formal_snapshot_always_reserves_all_public_development_roles(self):
        f=self.fixture()
        f.draws=iter([e.SMOKE_SEEDS[0]-1_000_000_000,10,11,12,13])
        prepare.prepare(f.stage,f.deadline,f.helper,run_kind=e.FORMAL)
        snapshot=json.loads((f.stage/'seed-inventory.local.json').read_text())
        doc=json.loads((f.stage/'private-seeds.local.json').read_text())
        self.assertTrue(set(e.SMOKE_SEEDS) <= set(snapshot['excluded_uint32']))
        self.assertFalse(set(e.SMOKE_SEEDS) & set(doc['seeds']))

    def test_smoke_requires_all_predeclared_roles_unused_before_stage(self):
        f=self.fixture()
        f.inventory=e.history.SeedInventory([*f.prior.private_values(),e.SMOKE_SEEDS[3]],1,1,f.parent)
        with self.assertRaises(ValueError):
            prepare.prepare(f.stage,f.deadline,f.helper,run_kind=e.SMOKE)
        self.assertFalse(f.stage.exists())
        self.assertNotIn('random',f.events)

    def test_smoke_uses_only_public_development_roles_and_is_explicitly_marked(self):
        f=self.fixture()
        receipt=prepare.prepare(f.stage,f.deadline,f.helper,run_kind=e.SMOKE)
        cfg=json.loads((f.stage/'registration.json').read_text())
        decision=json.loads((f.stage/'root-decision.json').read_text())
        document=json.loads((f.stage/'private-seeds.local.json').read_text())
        snapshot=json.loads((f.stage/'seed-inventory.local.json').read_text())
        self.assertEqual(document['seeds'],list(e.SMOKE_SEEDS))
        self.assertFalse(set(document['seeds']) & set(snapshot['excluded_uint32']))
        self.assertEqual(receipt['run_kind'],e.SMOKE)
        self.assertEqual(decision['run_kind'],e.SMOKE)
        e.require_run_kind(cfg,e.SMOKE)
        self.assertEqual(cfg['max_started_games'],1)
        self.assertEqual(cfg['selected_game_key'],e.SMOKE_KEY)
        self.assertEqual(cfg['official_limits'],e.SMOKE_OFFICIAL)
        self.assertNotIn('random',f.events)


class ModeIsolationTests(unittest.TestCase):
    def test_formal_and_development_contracts_reject_each_other(self):
        formal=e.run_contract(e.FORMAL); smoke=e.run_contract(e.SMOKE)
        e.require_run_kind(formal,e.FORMAL); e.require_run_kind(smoke,e.SMOKE)
        self.assertEqual(launch._entry_for(formal,e.FORMAL),launch.ENTRY)
        self.assertEqual(launch._entry_for(smoke,e.SMOKE),launch.SMOKE_ENTRY)
        for cfg,requested in ((formal,e.SMOKE),(smoke,e.FORMAL)):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError): e.require_run_kind(cfg,requested)
                with self.assertRaises(ValueError): launch._entry_for(cfg,requested)
        for invalid in ({**smoke,'max_started_games':True}, {**smoke,'selected_game_key':'other'},
                        {**smoke,'gates':{}}, {**smoke,'official_limits':{}}):
            with self.assertRaises(ValueError): e.require_run_kind(invalid,e.SMOKE)
            with self.assertRaises(ValueError): launch._entry_for(invalid,e.SMOKE)

    def test_formal_runner_refuses_smoke_before_preflight_or_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp)/'stage'; stage.mkdir()
            class Context:
                def __init__(self,*_): self.OUT=stage
                def refresh(self): return e.run_contract(e.SMOKE)
            with patch.object(runner.isolation,'validate_runtime'),patch.object(runner,'Context',Context), \
                 patch.object(runner,'EpochContext',side_effect=lambda value:value), \
                 patch.object(runner.memory_gate,'wait_for_preflight') as preflight:
                with self.assertRaises(ValueError): runner.run(stage,'a'*40,temp)
            preflight.assert_not_called()
            self.assertFalse((stage/'results').exists())

    def test_development_runner_refuses_formal_before_preflight_or_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp)/'stage'; stage.mkdir()
            class Context:
                def __init__(self,*_): self.OUT=stage
                def refresh(self): return e.run_contract(e.FORMAL)
            with patch.object(smoke_runner.isolation,'validate_runtime'),patch.object(smoke_runner,'Context',Context), \
                 patch.object(smoke_runner,'EpochContext',side_effect=lambda value:value), \
                 patch.object(smoke_runner.memory_gate,'wait_for_preflight') as preflight:
                with self.assertRaises(ValueError): smoke_runner.run(stage,'a'*40,temp)
            preflight.assert_not_called()
            self.assertFalse((stage/'results').exists())

    def test_bootstrap_routes_smoke_and_fails_invalid_stage_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            scratch=Path(temp); project=Path(__file__).resolve().parents[2]
            command=isolation.bootstrap_command(sys.executable,scratch,scratch/'cache-smoke',project,
                                                'learning.panel_eval_v3.smoke_runner')
            command += ['--out',str(scratch/'missing'),'--prereg','a'*40,'--scratch',str(scratch)]
            result=subprocess.run(command,cwd=project,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual((result.stdout,result.stderr),(b'',b''))
            self.assertFalse((scratch/'cache-smoke').exists())


class ClaimTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.stage=Path(self.temp.name)/'stage'; self.stage.mkdir()
        self.key=e.SMOKE_KEY
        bot=self.stage/'agents'/(self.key+'.py'); bot.parent.mkdir(); bot.write_bytes(b'synthetic bot')
        self.sha=games.core.sha(bot)
        self.item={'game':self.key,'bot_sha256':self.sha}
        save(self.stage/'job-inputs'/(self.key+'.json'),self.item)
        save(self.stage/'results'/'coordinator-started.json',
             dict(attempt=1,prereg_commit='a'*40,run_kind=e.SMOKE,retry_allowed=False))
        save(self.stage/'results'/'canonical'/(self.key+'.json'),
             dict(game=self.key,wrapper_sha256=self.sha,passed=True,launcher_reaped=True,
                  job_members_zero=True,isolated_cache_prefix_absent=True,retry_allowed=False))
        save(self.stage/'results'/'job-start'/(self.key+'.json'),
             dict(game=self.key,wrapper_sha256=self.sha,retry_allowed=False))
        stage=self.stage
        class Context:
            OUT=stage; prereg_commit='a'*40
            def write_new(self,path,value):
                with Path(path).open('x',encoding='utf-8') as stream: json.dump(value,stream)
        self.context=Context()

    def test_wrong_game_and_duplicate_claim_fail_closed(self):
        with patch.object(games,'validate_plan'):
            with self.assertRaises(ValueError): worker.claim_smoke_game(self.context,{},'wrong-game')
            worker.claim_smoke_game(self.context,{},self.key)
            with self.assertRaises(FileExistsError): worker.claim_smoke_game(self.context,{},self.key)

    def test_canonical_or_start_evidence_tampering_prevents_claim(self):
        path=self.stage/'results'/'canonical'/(self.key+'.json')
        value=json.loads(path.read_text()); value['job_members_zero']=False
        save(path,value)
        with patch.object(games,'validate_plan'),self.assertRaises(ValueError):
            worker.claim_smoke_game(self.context,{},self.key)
        self.assertFalse((self.stage/'results'/'smoke-worker-claim.json').exists())

    def test_canonical_mode_never_consumes_game_claim(self):
        cfg=e.run_contract(e.SMOKE)
        context=self.context
        context.refresh=lambda:cfg
        context.ready=lambda:cfg
        args=['worker','--out',str(self.stage),'--key',self.key,'--prereg','a'*40,
              '--scratch',self.temp.name,'--mode','canonical']
        with patch.object(sys,'argv',args),patch('learning.panel_eval_v3.isolation.validate_runtime'), \
             patch('learning.panel_eval_v3.experiment.Context',return_value=context), \
             patch('learning.panel_eval_v3.epoch.EpochContext',side_effect=lambda value:value), \
             patch('learning.panel_eval_v3.adapter.bind_context',return_value=nullcontext()), \
             patch('learning.panel_eval_v3.wrapper_verify.verify_wrapper') as verify:
            worker.main()
        verify.assert_called_once()
        self.assertFalse((self.stage/'results'/'smoke-worker-claim.json').exists())

    def test_game_mode_rejects_wrong_key_before_dispatch(self):
        context=self.context; context.refresh=lambda:e.run_contract(e.SMOKE)
        args=['worker','--out',str(self.stage),'--key','wrong-key','--prereg','a'*40,
              '--scratch',self.temp.name,'--mode','game']
        with patch.object(sys,'argv',args),patch('learning.panel_eval_v3.isolation.validate_runtime'), \
             patch('learning.panel_eval_v3.experiment.Context',return_value=context), \
             patch('learning.panel_eval_v3.epoch.EpochContext',side_effect=lambda value:value), \
             patch.object(games,'job') as job:
            with self.assertRaises(ValueError): worker.main()
        job.assert_not_called()
        self.assertFalse((self.stage/'results'/'smoke-worker-claim.json').exists())


class SmokeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.stage=Path(self.temp.name)/'stage'; self.stage.mkdir()
        self.scratch=Path(self.temp.name)/'scratch'; self.scratch.mkdir()
        self.cfg=dict(e.run_contract(e.SMOKE),deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat(),
                      official_job_supervision={'limit':1})
        stage=self.stage; cfg=self.cfg
        class Context:
            ROOT=runner.ROOT
            def __init__(self,_,commit): self.OUT=stage; self.prereg_commit=commit
            def refresh(self): return cfg
            def ready(self,*_): return cfg
            def deadline_check(self): pass
            def load_seeds(self,*_): return list(e.SMOKE_SEEDS)
            def write_new(self,path,value):
                path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('x',encoding='utf-8') as stream: json.dump(value,stream)
        self.Context=Context

    def base_patches(self):
        return (patch.object(smoke_runner.isolation,'validate_runtime'),
                patch.object(smoke_runner,'Context',self.Context),
                patch.object(smoke_runner,'EpochContext',side_effect=lambda value:value))

    def test_initial_resource_failure_stops_and_consumes_attempt(self):
        a,b,c=self.base_patches()
        with a,b,c, patch.object(smoke_runner.memory_gate,'wait_for_preflight',return_value=SimpleNamespace(passed=False)), \
             patch.object(campaign,'run') as full,patch.object(games,'job') as job:
            with self.assertRaises(RuntimeError): smoke_runner.run(self.stage,'a'*40,self.scratch)
            with self.assertRaises(ValueError): smoke_runner.run(self.stage,'a'*40,self.scratch)
        full.assert_not_called(); job.assert_not_called()
        self.assertTrue((self.stage/'results'/'coordinator-failure.json').is_file())
        self.assertFalse((self.stage/'results'/'coordinator-complete.json').exists())

    def test_failed_job_cleanup_prevents_acceptance_and_retry(self):
        source=self.stage/'sources'/'round2.py'; source.parent.mkdir(); source.write_bytes(b'synthetic rival')
        class Controller:
            cleanup_state='unknown_after_spawn_attempt'
            def __init__(self,*_): pass
            def memory_ready(self,cfg): return True
            def supervise_game(self,*_): return {'passed':False,'job_members_zero':False}
            def accept_job(self,*_): raise AssertionError('Failed job must not be accepted')
        def wrapper(out,plan,cfg):
            path=Path(out)/'agents'/(plan['game']+'.py'); path.parent.mkdir(); path.write_bytes(b'synthetic bot'); return path
        a,b,c=self.base_patches()
        with a,b,c, patch.object(smoke_runner.memory_gate,'wait_for_preflight',return_value=SimpleNamespace(passed=True)), \
             patch.object(smoke_runner,'Coordinator',Controller),patch.object(smoke_runner,'WindowsAPI'), \
             patch.object(smoke_runner.adapter,'bind_context',return_value=nullcontext()), \
             patch.object(campaign,'storage_ready'),patch.object(campaign,'marker'), \
             patch.object(campaign,'identity',return_value={'freeze.json':'a'*64}), \
             patch.object(games,'make_wrapper',side_effect=wrapper),patch.object(games,'task_identity',return_value=(None,'b'*64)), \
             patch.object(games,'validate_plan'),patch.object(campaign,'run') as full, \
             patch.object(independent_runner,'finalize') as evaluate:
            with self.assertRaises(RuntimeError): smoke_runner.run(self.stage,'a'*40,self.scratch)
            with self.assertRaises(ValueError): smoke_runner.run(self.stage,'a'*40,self.scratch)
        full.assert_not_called(); evaluate.assert_not_called()
        failure=json.loads((self.stage/'results'/'coordinator-failure.json').read_text())
        self.assertEqual(failure['cleanup_state'],'unknown_after_spawn_attempt')
        self.assertFalse((self.stage/'results'/'coordinator-complete.json').exists())

    def test_later_resource_failure_stops_before_wrapper_or_worker(self):
        class Controller:
            cleanup_state='not_started'
            def __init__(self,*_): pass
            def memory_ready(self,cfg): return False
        a,b,c=self.base_patches()
        with a,b,c, patch.object(smoke_runner.memory_gate,'wait_for_preflight',return_value=SimpleNamespace(passed=True)), \
             patch.object(smoke_runner,'Coordinator',Controller), \
             patch.object(smoke_runner.adapter,'bind_context',return_value=nullcontext()), \
             patch.object(games,'make_wrapper') as wrapper,patch.object(games,'job') as job:
            with self.assertRaises(RuntimeError): smoke_runner.run(self.stage,'a'*40,self.scratch)
        wrapper.assert_not_called(); job.assert_not_called()
        self.assertTrue((self.stage/'results'/'coordinator-failure.json').is_file())

    def test_synthetic_success_saves_only_development_status_and_hashes(self):
        source=self.stage/'sources'/'round2.py'; source.parent.mkdir(); source.write_bytes(b'synthetic rival')
        key=e.SMOKE_KEY
        ok=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                error_kind=None,cleanup_error_kind=None,stop_reason=None,exit_code=0)
        class Controller:
            cleanup_state='confirmed_empty'
            def __init__(self,context,*_): self.context=context
            def memory_ready(self,cfg): return True
            def supervise_game(self,*_):
                stage=self.context.OUT
                item=json.loads((stage/'job-inputs'/(key+'.json')).read_text())
                digest=item['bot_sha256']
                save(stage/'results'/'canonical'/(key+'.json'),dict(game=key,wrapper_sha256=digest,passed=True,
                    launcher_reaped=True,job_members_zero=True,isolated_cache_prefix_absent=True,retry_allowed=False))
                save(stage/'results'/'job-start'/(key+'.json'),dict(game=key,wrapper_sha256=digest,retry_allowed=False))
                save(stage/'results'/'canonical-post'/(key+'.json'),dict(game=key,wrapper_sha256=digest,passed=True,retry_allowed=False))
                save(stage/'results'/'smoke-worker-claim.json',dict(run_kind=e.SMOKE,game=key,wrapper_sha256=digest,
                                                                   attempt=1,retry_allowed=False))
                for name in ('complete','audit','storage'):
                    save(stage/'job-results'/(key+'.'+name+'.json'),{'synthetic':True})
                for path in (stage/'games'/key/'replay.json.gz',stage/'traces'/(key+'.jsonl.gz')):
                    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'synthetic data')
                return ok
            def accept_job(self,*_): return {'net_profit':'SYNTHETIC_PRIVATE_VALUE'}, {'official_transitions':719}
        def wrapper(out,plan,cfg):
            path=Path(out)/'agents'/(plan['game']+'.py'); path.parent.mkdir(); path.write_bytes(b'synthetic bot'); return path
        a,b,c=self.base_patches()
        with a,b,c, patch.object(smoke_runner.memory_gate,'wait_for_preflight',return_value=SimpleNamespace(passed=True)), \
             patch.object(smoke_runner,'Coordinator',Controller),patch.object(smoke_runner,'WindowsAPI'), \
             patch.object(smoke_runner.adapter,'bind_context',return_value=nullcontext()), \
             patch.object(campaign,'storage_ready'),patch.object(campaign,'marker'), \
             patch.object(campaign,'identity',return_value={'freeze.json':'a'*64}),patch.object(campaign,'check_identity'), \
             patch.object(games,'make_wrapper',side_effect=wrapper),patch.object(games,'task_identity',return_value=(None,'b'*64)), \
             patch.object(games,'validate_plan'),patch.object(campaign,'run') as full, \
             patch.object(independent_runner,'run') as extra,patch.object(independent_runner,'finalize') as evaluate, \
             patch.object(smoke_runner.evaluation,'evaluate') as statistics:
            smoke_runner.run(self.stage,'a'*40,self.scratch)
        full.assert_not_called(); extra.assert_not_called(); evaluate.assert_not_called(); statistics.assert_not_called()
        result=(self.stage/'results'/'coordinator-complete.json').read_text()
        document=json.loads(result)
        self.assertTrue(document['passed'])
        self.assertEqual(document['run_kind'],e.SMOKE)
        self.assertEqual((document['complete_games'],document['states'],document['action_turns'],
                          document['primary_replay_transitions']),(1,720,719,719))
        self.assertFalse(document['economic_evaluation_completed'])
        self.assertNotIn('SYNTHETIC_PRIVATE_VALUE',result)


if __name__ == '__main__':
    unittest.main()
