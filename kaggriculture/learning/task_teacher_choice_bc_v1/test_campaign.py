import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import numpy

exp=types.ModuleType('learning.task_teacher_choice_bc_v1.experiment')
exp.ROOT=Path(__file__).resolve().parents[2]; exp.OUT=exp.ROOT/'synthetic'
for name in ('ready','verify_commit','deadline_check','read','sha','write_new'): setattr(exp,name,lambda *args:None)
old=types.ModuleType('learning.task_action_observation_v1.campaign'); old.observe_processes=lambda *args:None
spec=importlib.util.spec_from_file_location('learning.task_teacher_choice_bc_v1._campaign_test',Path(__file__).with_name('campaign.py'))
c=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_teacher_choice_bc_v1.experiment':exp,'learning.task_action_observation_v1.campaign':old}): spec.loader.exec_module(c)


class CampaignTest(unittest.TestCase):
    def test_collection_seal_requires_job_cleanup(self):
        cfg=dict(collection_supervision={},source_manifest_sha256='source',sampling_algorithm='fixed')
        worker=dict(passed=True,collection_attempt=1,freeze_sha256='freeze',source_manifest_sha256='source',
                    sampling_algorithm='fixed',array_sha256='array',raw_reads=dict(files_opened=8,files_completed=8,raw_decode_passes_per_file=1))
        result=dict(passed=True,launcher_reaped=True,job_members_zero=True)
        for clean in (True,False):
            records={}
            def marker(out,name,value,state): state.update(created=True,complete=True)
            def sha(path): return {'freeze.json':'freeze','train.npz':'array','data-supervision-result.json':'job'}[Path(path).name]
            with patch.object(c,'ready',return_value=cfg),patch.object(c,'verify_commit'),patch.object(Path,'exists',return_value=False), \
                 patch.object(c,'marker',side_effect=marker),patch.object(c,'WindowsAPI',return_value=None), \
                 patch.object(c,'supervise',return_value={**result,'job_members_zero':clean}),patch.object(c,'read',return_value=worker), \
                 patch.object(c,'sha',side_effect=sha),patch.object(c,'deadline_check'), \
                 patch.object(c,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})):
                if clean:
                    self.assertTrue(c.collect_data(exp.OUT)['sealed'])
                    self.assertIn('train-data-result.json',records)
                else:
                    with self.assertRaises(c.CampaignFailure): c.collect_data(exp.OUT)
                    self.assertNotIn('train-data-result.json',records)

    def test_fit_rejects_unsealed_collection_before_marker(self):
        with patch.object(c,'ready',return_value={}),patch.object(c,'verify_commit'),patch.object(Path,'exists',return_value=False), \
             patch.object(c,'read',side_effect=[dict(passed=True,sealed=False),dict(passed=True)]),patch.object(c,'marker') as mark:
            with self.assertRaises(ValueError): c.fit(exp.OUT)
            mark.assert_not_called()

    def test_partial_data_marker_consumes_attempt(self):
        records={}
        def marker(out,name,value,state):
            records[name]=value; state['created']=True
            raise OSError('partial')
        with patch.object(c,'ready',return_value={}),patch.object(c,'verify_commit'), \
             patch.object(Path,'exists',autospec=True,side_effect=lambda p:Path(p).name in records), \
             patch.object(c,'sha',return_value='freeze'),patch.object(c,'marker',side_effect=marker), \
             patch.object(c,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})):
            with self.assertRaises(c.CampaignFailure): c.collect_data(exp.OUT)
            self.assertFalse(records['data-failure.json']['marker']['complete'])
            with self.assertRaises(ValueError): c.collect_data(exp.OUT)

    def test_partial_marker_receipt_write_failure_preserves_primary(self):
        def marker(out,name,value,state): state['created']=True; raise OSError('partial')
        with patch.object(c,'ready',return_value={}),patch.object(c,'verify_commit'),patch.object(Path,'exists',return_value=False), \
             patch.object(c,'sha',return_value='freeze'),patch.object(c,'marker',side_effect=marker), \
             patch.object(c,'write_new',side_effect=PermissionError('receipt')):
            with self.assertRaises(c.CampaignFailure) as caught: c.collect_data(exp.OUT)
        self.assertEqual(caught.exception.primary_error_kind,'OSError')
        self.assertEqual(caught.exception.receipt_error_kind,'PermissionError')

    def test_fit_transport_and_seal(self):
        for case in ('success','partial','spawn','windows','linux','checkpoint','data'):
            with self.subTest(case=case), ExitStack() as stack:
                records={}
                cfg=dict(source_manifest_sha256='source',sampling_algorithm='fixed',gpu_launch_config_sha256='launch',
                    storage=dict(process_log_bytes=100),training=dict(process_seconds=100))
                source=dict(passed=True,sealed=True,job_supervision_sha256='job',freeze_sha256='freeze',collection_attempt=1,
                    source_manifest_sha256='source',sampling_algorithm='fixed',array_sha256='array')
                guard=dict(passed=True,supervision=dict(owned_process_group_signalled=True,direct_child_reaped=True,
                    no_live_owned_session_members_before_reap=case!='linux'))
                summary=dict(synchronized_updates=128,gpu_verified=True,weights_changed=True,nonzero_gradient_updates=128,
                    inference_parity=dict(groups=16,synchronized=True,requested_matmul_precision='highest',hardware_precision_measured=False))
                fitted=dict(passed=True,updates=128,training_summary=summary,checkpoint_sha256='model',
                            train_data_sha256='changed' if case=='data' else 'data')
                def read(path):
                    return {'train-data-result.json':source,'data-supervision-result.json':dict(passed=True),
                        'gpu-launch.local.json':dict(distro='x',workdir='x',python='x'),
                        'gpu-guard-result.json':guard,'fit-result.json':fitted}[Path(path).name]
                def sha(path):
                    return {'data-supervision-result.json':'job','freeze.json':'freeze','train.npz':'array',
                        'train-data-result.json':'data','gpu-launch.local.json':'launch',
                        'model-final.npz':'wrong' if case=='checkpoint' else 'model',
                        'fit-result.json':'fit','fit-transport-result.json':'transport'}[Path(path).name]
                def marker(out,name,value,state):
                    records[name]=value; state['created']=True
                    if case=='partial': raise OSError('partial')
                    state['complete']=True
                class Process:
                    returncode=0
                    def poll(self): return 0
                class Child:
                    process=Process()
                    def finish(self): return dict(child_exit_code=0)
                for name,value in (('ready',cfg),('verify_commit',None),('deadline_check',None),('observe_processes',None)):
                    stack.enter_context(patch.object(c,name,return_value=value))
                stack.enter_context(patch.object(Path,'exists',autospec=True,side_effect=lambda p:Path(p).name in records or Path(p).name=='gpu-guard-result.json'))
                stack.enter_context(patch.object(c,'read',side_effect=read)); stack.enter_context(patch.object(c,'sha',side_effect=sha))
                stack.enter_context(patch.object(c,'marker',side_effect=marker))
                stack.enter_context(patch.object(c,'write_new',side_effect=lambda p,v:records.update({Path(p).name:v})))
                stack.enter_context(patch.object(c,'CapturedProcess',return_value=Child(),side_effect=OSError() if case=='spawn' else None))
                stack.enter_context(patch.object(c,'reap',return_value=dict(owned_instances_remaining=0,reader_joined=case!='windows',launched_process_reaped=True)))
                if case=='success': self.assertTrue(c.fit(exp.OUT)['passed'])
                else:
                    with self.assertRaises(c.CampaignFailure): c.fit(exp.OUT)
                    self.assertNotIn('model-seal.json',records)
                if case=='partial': self.assertFalse(records['fit-transport-result.json']['marker']['complete'])
                if case=='spawn': self.assertEqual(records['fit-transport-result.json']['child_lifecycle'],'unknown_after_spawn_attempt')
                if case in ('checkpoint','data'): self.assertIn('fit-seal-failure.json',records)
                with self.assertRaises(ValueError): c.fit(exp.OUT)


if __name__ == '__main__': unittest.main()
