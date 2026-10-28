import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy

exp=types.ModuleType('learning.task_teacher_refresh_v2.experiment')
exp.OUT=exp.ROOT=Path(__file__).resolve().parent/'synthetic'
for name in ('ready','verify_commit','deadline_check','read','sha','write_new'): setattr(exp,name,lambda *a:None)
games=types.ModuleType('learning.task_teacher_refresh_v2.games')
games.plans=lambda cfg:[dict(game=f'g{i}',seed=1+i//4,opponent=('round2' if i%4<2 else 'market'),side=i%2,
                            policy='teacher_task',split='dev',parity_kind=None) for i in range(8)]
games.validate_plan=lambda *a,**k:None
games.make_wrapper=lambda out,plan,cfg:out/'agents'/'wrapper.py'
games.validate_receipts=lambda *a:None
games.task_identity=lambda *a:(None,None)
spec=importlib.util.spec_from_file_location('learning.task_teacher_refresh_v2._campaign_test',Path(__file__).with_name('campaign.py'))
c=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_teacher_refresh_v2.experiment':exp,'learning.task_teacher_refresh_v2.games':games}): spec.loader.exec_module(c)


class Tests(unittest.TestCase):
    def run_fake(self,mode):
        cfg=dict(host_preflight=dict(samples=3,min_available_bytes=4*1024**3),
                 official_job_supervision=dict(min_available_bytes=2*1024**3))
        files={}; calls=[]; memory=[]
        def write(path,value):
            key=str(path)
            if mode=='receipt' and path.parent.name=='job-supervision': raise OSError('synthetic')
            files[key]=value
        def supervisor(*a):
            calls.append(1)
            if mode=='second_error' and len(calls)==2: raise OSError('synthetic')
            return dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,
                        error_kind=None,cleanup_error_kind=None,stop_reason=None,exit_code=0)
        def marker(out,state): state.update(created=True,complete=True)
        def available(cfg): memory.append(1); return mode!='preflight'
        def accept(out,item,cfg): return dict(game=item['game']),dict(game=item['game'])
        def verify(*args):
            if mode=='drift': raise ValueError('synthetic evidence changed')
        with patch.object(c,'ready',return_value=cfg),patch.object(c,'verify_commit'),patch.object(c,'deadline_check'), \
             patch.object(Path,'exists',return_value=False),patch.object(c,'memory_ready',side_effect=available), \
             patch.object(c,'marker',side_effect=marker),patch.object(c,'sha',return_value='hash'), \
             patch.object(c,'write_new',side_effect=write),patch.object(c,'read',side_effect=lambda p:files[str(p)]), \
             patch.object(c,'accept_job',side_effect=accept),patch.object(c,'verify_ledger',side_effect=verify), \
             patch.object(c,'make_manifest',return_value=[]),patch.object(c,'supervise',side_effect=supervisor), \
             patch.object(c,'WindowsAPI',return_value=object()):
            if mode=='success': self.assertTrue(c.run(exp.OUT)['passed'])
            else:
                with self.assertRaises((c.CampaignFailure,RuntimeError)): c.run(exp.OUT)
        return files,calls,memory

    def test_success_eight_jobs(self):
        files,calls,memory=self.run_fake('success')
        self.assertEqual((len(calls),len(memory)),(8,8))
        self.assertEqual(files[str(exp.OUT/'games-result.json')]['complete_games'],8)

    def test_receipt_failure_or_supervision_error_stops(self):
        files,calls,_=self.run_fake('receipt'); self.assertEqual(len(calls),1)
        files,calls,_=self.run_fake('second_error'); self.assertEqual(len(calls),2)
        first=games.plans({})[0]['game']; second=games.plans({})[1]['game']
        self.assertIn(str(exp.OUT/'job-acceptance'/(first+'.json')),files)
        self.assertEqual(files[str(exp.OUT/'job-supervision'/(second+'.json'))]['cleanup_state'],'unknown')
        self.assertNotIn(str(exp.OUT/'games-result.json'),files)

    def test_preflight_and_final_recheck_failure(self):
        files,calls,_=self.run_fake('preflight'); self.assertFalse(calls)
        files,calls,_=self.run_fake('drift'); self.assertEqual(len(calls),8)
        self.assertNotIn(str(exp.OUT/'games-result.json'),files)

    def test_manifest_maps_all_sources(self):
        audits=[dict(game=p['game'],trace_sha256='t',replay_sha256='r',receipt_sha256={'audit':'a'}) for p in games.plans({})]
        with patch.object(Path,'stat',return_value=types.SimpleNamespace(st_size=5)):
            result=c.make_manifest(exp.OUT,{},audits)
        self.assertEqual(len(result),8); self.assertEqual(result[2]['opponent'],'market')
        self.assertEqual(result[0]['trace']['size'],5); self.assertEqual(result[0]['audit_sha256'],'a')

    def test_real_recheck_catches_first_supervision_changed_in_second_job(self):
        out=exp.OUT; files={}; calls=[]
        cfg=dict(host_preflight={'samples':3,'min_available_bytes':4*1024**3},
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
        def supervisor(*args):
            calls.append(1)
            if len(calls)==2:
                key=games.plans(cfg)[0]['game']
                files[str(out/'job-supervision'/(key+'.json'))]['supervision']['exit_code']=7
            return copy.deepcopy(job)
        def accept(out,item,cfg): return ({'game':item['game']},{'game':item['game'],'receipt_sha256':{'audit':'fixed'}})
        with patch.object(c,'ready',return_value=cfg),patch.object(c,'verify_commit'),patch.object(c,'deadline_check'), \
             patch.object(Path,'exists',return_value=False),patch.object(c,'memory_ready',return_value=True), \
             patch.object(c,'marker',side_effect=lambda out,state:state.update(created=True,complete=True)), \
             patch.object(c,'sha',side_effect=sha),patch.object(c,'read',side_effect=read),patch.object(c,'write_new',side_effect=write), \
             patch.object(c,'accept_job',side_effect=accept),patch.object(c,'supervise',side_effect=supervisor), \
             patch.object(c,'WindowsAPI',return_value=object()):
            with self.assertRaises(c.CampaignFailure): c.run(out)
        self.assertEqual(len(calls),8); self.assertNotIn(str(out/'games-result.json'),files)
        self.assertEqual(files[str(out/'games-failure.local.json')]['completed_games'],8)

    def test_validate_panel_rejects_manifest_size_trace_and_audit_changes(self):
        out=exp.OUT; cfg={}; audit=[dict(game=p['game'],trace_sha256='t',replay_sha256='r',
            receipt_sha256={'audit':'a'}) for p in games.plans(cfg)]
        size=[5]
        def stat(): return types.SimpleNamespace(st_size=size[0])
        with patch.object(Path,'stat',side_effect=stat): original=c.make_manifest(out,cfg,audit)
        files={'games-result.json':dict(passed=True,complete_games=8,live_official_transitions=8*719,
            primary_replay_transitions=8*719,game_ledger_sha256='hash',audit_ledger_sha256='hash',
            source_manifest_sha256='hash',identities={}), 'game-ledger.json':[{}]*8,
            'game-audit-ledger.json':audit,'source-manifest.json':original}
        with patch.object(c,'ready'),patch.object(c,'verify_commit'),patch.object(c,'verify_ledger'), \
             patch.object(c,'sha',return_value='hash'),patch.object(c,'read',side_effect=lambda p:files[Path(p).name]), \
             patch.object(Path,'stat',side_effect=stat):
            self.assertEqual(len(c.validate_panel(out,cfg)),8)
            size[0]=6
            with self.assertRaises(ValueError): c.validate_panel(out,cfg)
            size[0]=5
            for key in ('trace_sha256','audit'):
                changed=copy.deepcopy(audit)
                if key=='audit': changed[0]['receipt_sha256']['audit']='changed'
                else: changed[0][key]='changed'
                files['game-audit-ledger.json']=changed
                with self.assertRaises(ValueError): c.validate_panel(out,cfg)


if __name__=='__main__': unittest.main()
