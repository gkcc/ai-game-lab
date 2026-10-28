import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy  # Keep NumPy's extension loaded when sys.modules is temporarily patched.
from .evaluation import plans

exp=types.ModuleType('learning.task_teacher_weighted_eval_v2.experiment')
exp.OUT=exp.ROOT=Path(__file__).resolve().parent/'synthetic'
for name in ('ready','verify_commit','deadline_check','read','sha','write_new'): setattr(exp,name,lambda *a:None)
games=types.ModuleType('learning.task_teacher_weighted_eval_v2.games')
games.plans=plans
games.validate_plan=lambda *a,**k:None
games.make_wrapper=lambda out,plan,cfg:out/'agents'/('wrapper.py')
games.validate_receipts=lambda *a:None
games.task_identity=lambda out,plan,cfg:(out/'model.npz','model')
resources=types.ModuleType('learning.task_teacher_weighted_eval_v2.resources')
resources.memory_ready=lambda cfg:True
spec=importlib.util.spec_from_file_location('learning.task_teacher_weighted_eval_v2._campaign_test',Path(__file__).with_name('campaign.py'))
c=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_teacher_weighted_eval_v2.experiment':exp,
    'learning.task_teacher_weighted_eval_v2.games':games,'learning.task_teacher_weighted_eval_v2.resources':resources}): spec.loader.exec_module(c)


class CampaignTests(unittest.TestCase):
    def run_fake(self,scenario):
        cfg=dict(test_seeds=[1,2,3],host_preflight={'samples':3,'min_available_bytes':4*1024**3},
                 official_job_supervision={'min_available_bytes':2*1024**3})
        writes={}; calls=[]; mem=[]; version=['same']
        def write(path,value):
            if scenario=='receipt_failure' and Path(path).parent.name=='job-supervision': raise OSError('receipt')
            writes[str(Path(path).parent.name)+'/'+Path(path).name if Path(path).parent.name in ('job-supervision','job-acceptance') else Path(path).name]=value
        def memory(_):
            mem.append(1)
            return not (scenario=='preflight' or (scenario=='midmemory' and len(mem)>1))
        def marker(out,state):
            state['created']=True
            if scenario=='partial': raise OSError('partial')
            state['complete']=True
        def supervisor(*args):
            calls.append(1)
            if scenario=='drift': version[0]='changed'
            if scenario=='spawn' or (scenario=='second_spawn' and len(calls)==2): raise OSError('spawn')
            return dict(passed=scenario!='job',assigned=True,resumed=True,launcher_reaped=True,
                        job_members_zero=True,error_kind=None,cleanup_error_kind=None,stop_reason=None,exit_code=0)
        def sha(path):
            name=Path(path).name
            return version[0] if name=='freeze.json' else 'hash'
        def accept(out,item,cfg): return ({'game':item['game']},{'game':item['game']})
        with patch.object(c,'ready',return_value=cfg),patch.object(c,'verify_commit'),patch.object(c,'deadline_check'), \
             patch.object(Path,'exists',return_value=False),patch.object(c,'memory_ready',side_effect=memory), \
             patch.object(c,'marker',side_effect=marker),patch.object(c,'sha',side_effect=sha), \
             patch.object(c,'write_new',side_effect=write),patch.object(c,'accept_job',side_effect=accept), \
             patch.object(c,'supervise',side_effect=supervisor),patch.object(c,'WindowsAPI',return_value=object()), \
             patch.object(c,'verify_ledger'):
            if scenario=='success': self.assertTrue(c.run(exp.OUT)['passed'])
            else:
                with self.assertRaises((c.CampaignFailure,RuntimeError)): c.run(exp.OUT)
        return writes,calls,mem

    def test_preflight_stops_before_marker(self):
        writes,calls,_=self.run_fake('preflight')
        self.assertEqual(calls,[]); self.assertNotIn('games-failure.local.json',writes)

    def test_partial_marker_records_failure(self):
        writes,calls,_=self.run_fake('partial')
        self.assertEqual(calls,[]); self.assertFalse(writes['games-failure.local.json']['marker']['complete'])

    def test_midpanel_memory(self):
        writes,calls,_=self.run_fake('midmemory'); failure=writes['games-failure.local.json']
        self.assertEqual(len(calls),1); self.assertEqual(failure['completed_games'],1)
        self.assertEqual(failure['current_game_additional_transitions'],0)
        self.assertIsNone(failure['uncertain_current_game'])
        self.assertEqual(failure['stop_reason'],'available_memory_preflight')

    def test_job_failure(self):
        writes,calls,_=self.run_fake('job'); failure=writes['games-failure.local.json']
        self.assertEqual(len(calls),1); self.assertEqual(failure['current_game_additional_transitions'],'unknown')
        self.assertEqual(failure['uncertain_current_game'],plans([1,2,3])[0]['game'])
        self.assertFalse(failure['retry_allowed'])

    def test_spawn_failure(self):
        writes,calls,_=self.run_fake('spawn')
        self.assertEqual(writes['games-failure.local.json']['child_lifecycle'],'unknown_after_spawn_attempt')
        self.assertEqual(writes['games-failure.local.json']['uncertain_current_game'],plans([1,2,3])[0]['game'])

    def test_identity_drift(self):
        writes,calls,_=self.run_fake('drift')
        self.assertEqual(len(calls),1); self.assertNotIn('games-result.json',writes)

    def test_success_24_games(self):
        writes,calls,mem=self.run_fake('success')
        self.assertEqual(len(calls),24); self.assertEqual(len(mem),24)
        self.assertEqual(writes['games-result.json']['live_official_transitions'],24*719)
        self.assertTrue(writes['games-result.json']['independent_replay_not_yet_completed'])

    def test_job_cleanup_flags(self):
        self.assertFalse(c.job_ok({'passed':True,'launcher_reaped':True,'job_members_zero':False}))

    def test_prior_receipts_survive_later_supervision_exception(self):
        writes,calls,_=self.run_fake('second_spawn'); self.assertEqual(len(calls),2)
        self.assertEqual(len([k for k in writes if k.startswith('job-supervision/')]),2)
        self.assertEqual(len([k for k in writes if k.startswith('job-acceptance/')]),1)
        second=plans([1,2,3])[1]['game']
        self.assertEqual(writes['job-supervision/'+second+'.json']['cleanup_state'],'unknown')

    def test_receipt_write_failure_stops_panel(self):
        writes,calls,_=self.run_fake('receipt_failure')
        self.assertEqual(len(calls),1); self.assertNotIn('games-result.json',writes)

    def test_previous_receipt_changed_during_second_game_blocks_success(self):
        out=exp.OUT; files={}; calls=[]
        cfg=dict(test_seeds=[1,2,3],host_preflight={'samples':3,'min_available_bytes':4*1024**3},
                 official_job_supervision={'min_available_bytes':2*1024**3})
        job=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                 error_kind=None,cleanup_error_kind=None,stop_reason=None,exit_code=0)
        def write(path,value):
            key=str(path)
            if key in files: raise FileExistsError(key)
            files[key]=copy.deepcopy(value)
        def read(path): return copy.deepcopy(files[str(path)])
        def sha(path):
            value=files.get(str(path))
            return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest() if value is not None else 'fixed'
        def marker(out,state): state.update(created=True,complete=True)
        def supervisor(*args):
            calls.append(1)
            if len(calls)==2:
                first=plans([1,2,3])[0]['game']
                files[str(out/'job-supervision'/(first+'.json'))]['supervision']['exit_code']=7
            return copy.deepcopy(job)
        def accept(out,item,cfg):
            return ({'game':item['game']}, {'game':item['game'],'receipt_sha256':{'audit':'fixed'}})
        with patch.object(c,'ready',return_value=cfg),patch.object(c,'verify_commit'),patch.object(c,'deadline_check'), \
             patch.object(Path,'exists',return_value=False),patch.object(c,'memory_ready',return_value=True), \
             patch.object(c,'marker',side_effect=marker),patch.object(c,'sha',side_effect=sha), \
             patch.object(c,'read',side_effect=read),patch.object(c,'write_new',side_effect=write), \
             patch.object(c,'accept_job',side_effect=accept),patch.object(c,'supervise',side_effect=supervisor), \
             patch.object(c,'WindowsAPI',return_value=object()):
            with self.assertRaises(c.CampaignFailure): c.run(out)
        self.assertEqual(len(calls),24)
        self.assertNotIn(str(out/'games-result.json'),files)
        self.assertEqual(files[str(out/'games-failure.local.json')]['completed_games'],24)

    def test_validate_panel_rejects_changed_audit_receipt_hash(self):
        queue=plans([1,2,3]); files={}; hashes={}; out=exp.OUT
        identities={name:'identity' for name in ('freeze.json','registration.json','pre-execution-commit.json','games-started.json')}
        job=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                 error_kind=None,cleanup_error_kind=None,stop_reason=None,exit_code=0)
        fields=dict(passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
            bank=10,opponent_bank=10,initial_bank=10,opponent_initial_bank=10,net_profit=0,
            opponent_net_profit=0,margin=0,score=.5)
        rows=[]; audits=[]
        for p in queue:
            key=p['game']; item=dict(p,bot_sha256='hash',rival_sha256='hash',task_checkpoint_sha256='model')
            files[str(out/'job-inputs'/(key+'.json'))]=item
            row=dict(item,**fields,replay_gzip_sha256='hash',trace_sha256='hash')
            for name,value in [('row',row),('audit',{}),('complete',{}),('storage',{})]:
                path=out/'job-results'/(key+'.'+name+'.json'); files[str(path)]=value; hashes[str(path)]='hash'
            rows.append(c.sanitized(row))
            audit=dict(game=key,receipt_sha256={k:'hash' for k in ('row','audit','complete','storage')},
                replay_sha256='hash',trace_sha256='hash',task_checkpoint_sha256='model',official_transitions=719,job_supervision=job)
            files[str(out/'job-supervision'/(key+'.json'))]=dict(game=key,job_input_sha256='hash',freeze_sha256='identity',
                supervision=job,error_kind=None,cleanup_state='recorded',retry_allowed=False)
            files[str(out/'job-acceptance'/(key+'.json'))]=dict(passed=True,game=key,freeze_sha256='identity',
                job_input_sha256='hash',supervision_sha256='hash',evidence=audit.copy(),retry_allowed=False)
            audits.append(dict(audit,supervision_receipt_sha256='hash',acceptance_receipt_sha256='hash'))
        cfg=dict(test_seeds=[1,2,3],crop_checkpoint='crop.npz',crop_checkpoint_sha256='hash',
                 source_sha256={'round2.py':'hash','market.py':'hash'})
        files[str(out/'game-ledger.json')]=rows; files[str(out/'game-audit-ledger.json')]=audits
        files[str(out/'games-result.json')]=dict(passed=True,complete_games=24,live_official_transitions=24*719,
            primary_replay_transitions=24*719,game_ledger_sha256='hash',audit_ledger_sha256='hash',identities=identities)
        def sha(path):
            name=Path(path).name
            if name in identities: return 'identity'
            if name=='model.npz': return 'model'
            return hashes.get(str(path),'hash')
        with patch.object(c,'ready'),patch.object(c,'verify_commit'),patch.object(c,'read',side_effect=lambda p:files[str(p)]), \
             patch.object(c,'sha',side_effect=sha),patch.object(c,'verify_economy'),patch.object(c,'validate_receipts'):
            self.assertEqual(len(c.validate_panel(out,cfg)),24)
            hashes[str(out/'job-results'/(queue[0]['game']+'.audit.json'))]='changed'
            with self.assertRaises(ValueError): c.validate_panel(out,cfg)
            hashes[str(out/'job-results'/(queue[0]['game']+'.audit.json'))]='hash'
            hashes[str(out/'job-supervision'/(queue[0]['game']+'.json'))]='changed'
            with self.assertRaises(ValueError): c.validate_panel(out,cfg)


if __name__=='__main__': unittest.main()
