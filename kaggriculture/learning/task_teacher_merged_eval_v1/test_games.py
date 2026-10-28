import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

SEEDS=[3112345671,3112345682,3112345693,3112345704]
core=types.ModuleType('learning.task_anchor_once_v1.games')
core.verify,core.task_identity,core.read,core.write_new=object(),object(),lambda _:None,lambda *_:None
exp=types.ModuleType('learning.task_teacher_merged_eval_v1.experiment')
exp.ROOT=Path(__file__).resolve().parents[2]
exp.OUT=exp.ROOT/'synthetic'
exp.verify=lambda *args:None
exp.ready=lambda *args:None
exp.verify_commit=lambda *args:None
exp.deadline_check=lambda:None
exp.load_seeds=lambda *args:list(SEEDS)
spec=importlib.util.spec_from_file_location('learning.task_teacher_merged_eval_v1._games_test',Path(__file__).with_name('games.py'))
g=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_anchor_once_v1.games':core,
                             'learning.task_teacher_merged_eval_v1.experiment':exp}):
    spec.loader.exec_module(g)
# A previously imported package may retain its real ``experiment`` attribute
# even while the matching sys.modules entry is patched. Bind the explicit fakes
# to the dynamically loaded adapter so suite order cannot change its behavior.
g.experiment, g.core = exp, core


class GamesTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(merged_checkpoint='new/model.npz',merged_checkpoint_sha256='a'*64,
            unweighted_checkpoint='old/model.npz',unweighted_checkpoint_sha256='b'*64,
            crop_checkpoint='crop/model.npz',source_sha256={'round2.py':'d'*64,'market.py':'c'*64,'teacher.py':'f'*64},
            engine_sha256='e'*64,crop_checkpoint_sha256='9'*64,
            storage=dict(game_write_bytes=128*1024*1024),
            game_limits=dict(controller_calls=719,task_forwards=4000,crop_forwards=8000,neural_input_rows=160000,
                             gate_evaluations=1,teacher_proposals=1))

    def job_plan(self,policy='merged',opponent='round2'):
        p=next(p for p in g.plans(SEEDS) if p['policy']==policy and p['opponent']==opponent)
        model={'merged':'a'*64,'unweighted':'b'*64,'teacher_task':None}[policy]
        return dict(p,bot_sha256='8'*64,rival_sha256=self.cfg['source_sha256'][opponent+'.py'],task_checkpoint_sha256=model)

    def receipts(self,plan):
        row=dict(plan,passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
            engine_sha256='e'*64,crop_checkpoint_sha256='9'*64,replay_gzip_sha256='1'*64,trace_sha256='2'*64,
            semantic_replay_sha256='3'*64,semantic_replay_sha256_v2='4'*64,rewards=[1,2])
        forwards=0 if plan['policy']=='teacher_task' else 10
        counts=dict(task_forwards=forwards,task_rows=forwards*2,crop_forwards=3,crop_rows=3,
                    gate_evaluations=0,controller_calls=719,teacher_proposals=0)
        audit=dict(game=plan['game'],passed=True,errors=0,actor_actions_reproduced=719,rival_actions_reproduced=719,
            trace_turns_reproduced=719,replay_sha256='1'*64,trace_sha256='2'*64,
            live_model_counts=counts,primary_model_counts=dict(counts),official=dict(official_transitions_reexecuted=719,
            passed=True,engine_sha256='e'*64,semantic_sha256='3'*64,semantic_sha256_v2='4'*64,final_rewards=[1,2],
            both_public_and_private_states_match_each_step=True,inventory_order_checked_each_step=True))
        complete=dict(game=plan['game'],passed=True,row_sha256='5'*64,certificate_sha256='6'*64)
        limit=self.cfg['storage']['game_write_bytes']
        storage=dict(game=plan['game'],passed=True,error_kind=None,
                     output=dict(limit_bytes=limit,used_bytes=1024,remaining_bytes=limit-1024,created_count=4,refused=0),
                     native_output_accounted_by=g.NATIVE_OUTPUT_SCOPE)
        return row,audit,complete,storage

    def test_three_model_branches_and_paths(self):
        for plan in g.plans(SEEDS):
            g.validate_plan(plan,self.cfg)
            _,actual=g.task_identity(None,plan,self.cfg)
            self.assertEqual(actual,{'merged':'a'*64,'unweighted':'b'*64,'teacher_task':None}[plan['policy']])
        plan=g.plans(SEEDS)[0]
        for change in ({'seed':1},{'seed_alias':'S02'},{'side':True},{'opponent':'unknown'},
                       {'policy':'other'},{'game':'changed'}):
            with self.assertRaises(ValueError): g.validate_plan({**plan,**change},self.cfg)
        for path in ('../outside','/absolute','x/../other','C:'+'/bad','x\\bad'):
            with self.assertRaises(ValueError): g.task_identity(None,plan,{**self.cfg,'merged_checkpoint':path})
        with self.assertRaises(ValueError): g.validate_plan({**self.job_plan(),'task_checkpoint_sha256':None},self.cfg,job=True)
        teacher=self.job_plan('teacher_task','teacher')
        with self.assertRaises(ValueError): g.validate_plan({**teacher,'task_checkpoint_sha256':'a'*64},self.cfg,job=True)

    def test_receipts_distinguish_model_and_teacher(self):
        for policy in ('merged','unweighted','teacher_task'):
            plan=self.job_plan(policy,'teacher'); row,audit,complete,storage=self.receipts(plan)
            g.validate_receipts(plan,row,audit,complete,storage,'5'*64,'6'*64,self.cfg)
            for bad in ({**row,'agent_errors':1},{**row,'trace_sha256':'invalid'}):
                with self.assertRaises(ValueError): g.validate_receipts(plan,bad,audit,complete,storage,'5'*64,'6'*64,self.cfg)
            for bad in ({**audit,'errors':1},{**audit,'trace_sha256':'7'*64},
                        {**audit,'official':{**audit['official'],'passed':False}},
                        {**audit,'official':{**audit['official'],'inventory_order_checked_each_step':False}},
                        {**audit,'primary_model_counts':{**audit['primary_model_counts'],'task_rows':21}}):
                with self.assertRaises(ValueError): g.validate_receipts(plan,row,bad,complete,storage,'5'*64,'6'*64,self.cfg)
            counts=dict(audit['live_model_counts'])
            counts['task_forwards']=1 if policy=='teacher_task' else 0
            counts['task_rows']=1 if policy=='teacher_task' else 0
            bad={**audit,'live_model_counts':counts,'primary_model_counts':dict(counts)}
            with self.assertRaises(ValueError): g.validate_receipts(plan,row,bad,complete,storage,'5'*64,'6'*64,self.cfg)
            for bad in ({**storage,'native_output_accounted_by':'bounded parent process log'},
                        {**storage,'native_output_accounted_by':None},{**storage,'game':'different'}):
                with self.assertRaises(ValueError):
                    g.validate_receipts(plan,row,audit,complete,bad,'5'*64,'6'*64,self.cfg)
            output=storage['output']; limit=output['limit_bytes']
            for bad_output in ({key:value for key,value in output.items() if key!='limit_bytes'},
                               {key:value for key,value in output.items() if key!='created_count'},
                               {**output,'limit_bytes':limit+1}, {**output,'used_bytes':limit+1,'remaining_bytes':0},
                               {**output,'remaining_bytes':output['remaining_bytes']+1},
                               {**output,'created_count':-1},{**output,'refused':1},
                               *[{**output,key:True} for key in output]):
                with self.assertRaises(ValueError):
                    g.validate_receipts(plan,row,audit,complete,{**storage,'output':bad_output},'5'*64,'6'*64,self.cfg)

    def test_changed_second_read_restores_bindings(self):
        plan=self.job_plan(); changed={**plan,'bot_sha256':'1'*64}
        out=exp.OUT; path=(out/'job-inputs'/(plan['game']+'.json')).resolve()
        original=(core.verify,core.task_identity,core.read,core.write_new)
        with patch.object(exp,'ready',return_value=self.cfg),patch.object(core,'job',lambda *_:core.read(path),create=True), \
             patch.object(Path,'open',side_effect=[io.StringIO(json.dumps(plan)),io.StringIO(json.dumps(changed))]):
            with self.assertRaises(ValueError): g.job(out,plan['game'])
        self.assertEqual((core.verify,core.task_identity,core.read,core.write_new),original)

    def test_worker_discards_python_stdout_and_stderr(self):
        plan=self.job_plan(); out=exp.OUT; path=(out/'job-inputs'/(plan['game']+'.json')).resolve()
        def noisy_job(*_):
            print('private seed',plan['seed']); print('private seed',plan['seed'],file=sys.stderr)
            return core.read(path)
        stdout,stderr=io.StringIO(),io.StringIO()
        with patch.object(exp,'ready',return_value=self.cfg),patch.object(core,'job',noisy_job,create=True), \
             patch.object(Path,'open',side_effect=[io.StringIO(json.dumps(plan)),io.StringIO(json.dumps(plan))]), \
             redirect_stdout(stdout),redirect_stderr(stderr):
            self.assertEqual(g.job(out,plan['game']),plan)
        self.assertEqual(stdout.getvalue(),''); self.assertEqual(stderr.getvalue(),'')

    def test_module_cli_hides_exception_and_parser_output_in_subprocess(self):
        secret='synthetic-secret-for-cli-regression'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); flag=root/'called.txt'; out=root/'out'
            (out/'job-inputs').mkdir(parents=True)
            plan=self.job_plan(); (out/'job-inputs'/(plan['game']+'.json')).write_text(json.dumps(plan),encoding='utf-8')
            (root/'config.json').write_text(json.dumps(self.cfg),encoding='utf-8')
            # sitecustomize changes only this synthetic child process.  The
            # actual -m entry, parser and worker remain the production code.
            (root/'sitecustomize.py').write_text('''\
import json, os
from pathlib import Path
from learning.task_teacher_merged_eval_v1 import experiment
flag=Path(os.environ['KAGGRI_TEST_FLAG'])
secret=os.environ['KAGGRI_TEST_SECRET']
if os.environ['KAGGRI_TEST_CASE']=='failure':
    def ready(_):
        flag.write_text('ready reached',encoding='utf-8')
        raise ValueError(secret)
    experiment.ready=ready
else:
    from learning.task_anchor_once_v1 import games as core
    experiment.ROOT=Path(os.environ['KAGGRI_TEST_ROOT'])
    experiment.ready=lambda _:json.loads((experiment.ROOT/'config.json').read_text(encoding='utf-8'))
    experiment.verify_commit=lambda _:None
    experiment.deadline_check=lambda:None
    experiment.load_seeds=lambda *_:[3112345671,3112345682,3112345693,3112345704]
    def fake_job(*_):
        flag.write_text('job reached',encoding='utf-8')
        print(secret)
        print(secret,file=__import__('sys').stderr)
    core.job=fake_job
''',encoding='utf-8')
            env=os.environ.copy()
            env.update(KAGGRI_TEST_FLAG=str(flag),KAGGRI_TEST_SECRET=secret,KAGGRI_TEST_ROOT=str(root),
                       PYTHONPATH=os.pathsep.join((str(root),str(Path(__file__).resolve().parents[2]),env.get('PYTHONPATH',''))))
            command=[sys.executable,'-B','-m','learning.task_teacher_merged_eval_v1.games',
                     '--out',str(out),'--key',plan['game']]
            env['KAGGRI_TEST_CASE']='failure'
            failed=subprocess.run(command,cwd=Path(__file__).resolve().parents[2],env=env,
                                  capture_output=True,text=True,timeout=30)
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual(flag.read_text(encoding='utf-8'),'ready reached')
            self.assertEqual((failed.stdout,failed.stderr),('',''))
            flag.unlink()
            parser_error=subprocess.run(command+['--invalid',secret],cwd=Path(__file__).resolve().parents[2],env=env,
                                        capture_output=True,text=True,timeout=30)
            self.assertNotEqual(parser_error.returncode,0)
            self.assertFalse(flag.exists())
            self.assertEqual((parser_error.stdout,parser_error.stderr),('',''))
            env['KAGGRI_TEST_CASE']='success'
            succeeded=subprocess.run(command,cwd=Path(__file__).resolve().parents[2],env=env,
                                     capture_output=True,text=True,timeout=30)
            self.assertEqual(succeeded.returncode,0)
            self.assertEqual(flag.read_text(encoding='utf-8'),'job reached')
            self.assertEqual((succeeded.stdout,succeeded.stderr),('',''))

    def test_storage_adapter_is_per_game_and_restores_writer(self):
        plan=self.job_plan(); out=exp.OUT.resolve(); writes=[]
        path=out/'job-inputs'/(plan['game']+'.json')
        storage=out/'job-results'/(plan['game']+'.storage.json')
        original=dict(game=plan['game'],native_output_accounted_by='bounded parent process log')
        def writer(target,value): writes.append((Path(target),value))
        with patch.object(core,'write_new',writer):
            with g.bindings(path,plan,self.cfg,out):
                core.write_new(storage,original)
                core.write_new(out/'job-results'/(plan['game']+'.row.json'),{'value':1})
                for target,value in ((storage,{**original,'game':'different'}),
                                     (storage,{**original,'native_output_accounted_by':'unknown'}),
                                     (out/'job-results'/'different.storage.json',original)):
                    with self.assertRaises(ValueError): core.write_new(target,value)
            self.assertIs(core.write_new,writer)
        self.assertEqual(original['native_output_accounted_by'],'bounded parent process log')
        self.assertEqual(writes,[(storage,{**original,'native_output_accounted_by':g.NATIVE_OUTPUT_SCOPE}),
                                 (out/'job-results'/(plan['game']+'.row.json'),{'value':1})])

    def test_core_job_emits_scoped_storage_through_official_python_entry(self):
        """Synthetic entry and audit; actual core job, receipt writer, and consumer."""
        from learning.task_anchor_once_v1 import games as actual_core
        import run_match
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'experiment'
            for sub in ('agents','sources','job-inputs','traces','job-results'):
                (out/sub).mkdir(parents=True)
            plan=self.job_plan(); key=plan['game']
            bot=out/'agents'/(key+'.py'); rival=out/'sources'/(plan['opponent']+'.py')
            bot.write_bytes(b'synthetic bot'); rival.write_bytes(b'synthetic opponent')
            cfg=dict(self.cfg,engine_commit='f'*40,
                     storage=dict(game_write_bytes=2*1024*1024,small_receipt_bytes=16384))
            cfg['source_sha256']={**self.cfg['source_sha256'],rival.name:hashlib.sha256(rival.read_bytes()).hexdigest()}
            plan.update(bot_sha256=hashlib.sha256(bot.read_bytes()).hexdigest(),
                        rival_sha256=cfg['source_sha256'][rival.name])
            (out/'job-inputs'/(key+'.json')).write_text(json.dumps(plan),encoding='utf-8')
            invoked=[]; original_writer=actual_core.write_new; native=run_match.native_output
            def fake_entry():
                invoked.append(True)
                self.assertEqual(sys.argv[sys.argv.index('--seed')+1],str(plan['seed']))
                with run_match.native_output(None):
                    print('synthetic private seed',plan['seed'])
                    print('synthetic private seed',plan['seed'],file=sys.stderr)
                folder=out/'games'/key; folder.mkdir(parents=True)
                initial={'farms':[{'money':10000},{'money':10000}]}
                final={'farms':[{'money':10012},{'money':10001}]}
                replay=dict(info=dict(seed=plan['seed']),steps=[
                    [{'observation':initial if i==0 else final},{}] for i in range(720)])
                result=dict(passed=True,done=True,statuses=['DONE','DONE'],recorded_states=720,
                    action_turns=719,agent_errors=[],engine_matches_official_commit=True,
                    engine_sha256=cfg['engine_sha256'],upstream_commit=cfg['engine_commit'],
                    agent_file_sha256={str(bot):plan['bot_sha256'],str(rival):plan['rival_sha256']},
                    seed=plan['seed'],bot_side=plan['side'],rewards=[10012,10001],final_money=[10012,10001],
                    semantic_replay_sha256='3'*64,semantic_replay_sha256_v2='4'*64)
                (folder/'replay.json').write_text(json.dumps(replay),encoding='utf-8')
                (folder/'result.json').write_text(json.dumps(result),encoding='utf-8')
                (out/'traces'/(key+'.jsonl.gz')).write_bytes(b'synthetic trace')
                return 0
            def fake_audit(_out,row,_cfg):
                _,audit,_,_=self.receipts(plan)
                audit.update(replay_sha256=row['replay_gzip_sha256'],trace_sha256=row['trace_sha256'])
                audit['official']['final_rewards']=row['rewards']
                return audit
            stdout,stderr=io.StringIO(),io.StringIO()
            with patch.object(g,'core',actual_core),patch.object(actual_core,'ROOT',root), \
                 patch.object(actual_core,'audit',fake_audit),patch.object(exp,'ROOT',root), \
                 patch.object(exp,'OUT',out),patch.object(exp,'verify',return_value=cfg), \
                 patch.object(exp,'ready',return_value=cfg),patch.object(run_match,'__file__',str(root/'run_match.py')), \
                 patch.object(run_match,'main',fake_entry),redirect_stdout(stdout),redirect_stderr(stderr):
                g.job(out,key)
                self.assertIs(actual_core.write_new,original_writer)
                self.assertIs(run_match.native_output,native)
                def load(suffix): return json.loads((out/'job-results'/(key+suffix+'.json')).read_text(encoding='utf-8'))
                row,audit,complete,storage=(load(name) for name in ('.row','.audit','.complete','.storage'))
                self.assertEqual(storage['native_output_accounted_by'],g.NATIVE_OUTPUT_SCOPE)
                g.validate_receipts(plan,row,audit,complete,storage,complete['row_sha256'],
                                    complete['certificate_sha256'],cfg,out=out)
                with self.assertRaises(ValueError):
                    g.validate_receipts(plan,row,audit,complete,
                        {**storage,'native_output_accounted_by':'bounded parent process log'},
                        complete['row_sha256'],complete['certificate_sha256'],cfg,out=out)
            self.assertEqual(invoked,[True])
            self.assertEqual(stdout.getvalue(),''); self.assertEqual(stderr.getvalue(),'')

    def test_real_core_wrapper_uses_alias_not_seed(self):
        from learning.task_anchor_once_v1 import games as actual_core
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'experiment'; out.mkdir()
            plan=g.plans(SEEDS)[0]
            with patch.object(g,'core',actual_core),patch.object(actual_core,'ROOT',root),patch.object(exp,'ROOT',root):
                wrapper=g.make_wrapper(out,plan,self.cfg)
                text=wrapper.read_text(encoding='utf-8')
                self.assertNotIn(str(plan['seed']),wrapper.name)
                self.assertNotIn(str(plan['seed']),text)
                self.assertIn(plan['seed_alias'],wrapper.name)


if __name__=='__main__': unittest.main()
