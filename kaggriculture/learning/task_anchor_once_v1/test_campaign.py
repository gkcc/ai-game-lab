"""Stop boundaries, full registered panel reservation, and honest failure receipts."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import campaign
from .experiment import STORAGE, LIMITS
from .test_storage import _scratch


def config():
    return dict(storage=deepcopy(STORAGE),validation_seeds=[101,102],
        test_seeds=list(range(201,209)),opponents=['round2','market'],seats=[0,1],
        arms=['baseline','candidate','teacher_task'],game_limits=LIMITS,
        max_games=120,task_forward_budget=640000,crop_forward_budget=1920000,
        neural_input_row_budget=38400000,policy_turn_budget=345120,
        campaign_seconds=10800)


class CampaignTests(unittest.TestCase):
    def test_cancel_write_failure_still_waits_for_supervisor_and_joins_reader(self):
        calls=[]
        process=SimpleNamespace(pid=123,returncode=None,poll=lambda:None,
            wait=lambda **kwargs:calls.append(('wait',kwargs['timeout'])))
        thread=SimpleNamespace(join=lambda **kwargs:calls.append(('join',kwargs['timeout'])),
            is_alive=lambda:False)
        child=SimpleNamespace(process=process,thread=thread)
        cfg=config();cfg['gpu_guard']={'process_seconds':1200}
        runner=campaign.Runner(campaign.ROOT/'synthetic-not-created',cfg,None)
        def writer(path,value):
            if path.name=='training-cancel.local.json':raise OSError('Synthetic cancellation write refusal')
        with patch.object(campaign,'ready'),patch.object(runner,'storage_ready'), \
             patch.object(campaign,'sha',return_value='synthetic'), \
             patch.object(campaign,'read',return_value=dict(distro='synthetic',python='synthetic',workdir='synthetic')), \
             patch.object(campaign,'CapturedProcess',return_value=child), \
             patch.object(campaign,'write_new',side_effect=writer), \
             patch.object(runner,'limits',side_effect=RuntimeError('Synthetic parent interruption')):
            with self.assertRaisesRegex(OSError,'cancellation'):runner.gpu_fit()
        self.assertEqual([name for name,value in calls],['wait','join'])
        self.assertTrue(0<calls[0][1]<=1200)

    def test_free_floor_covers_parent_and_receipts_with_all_inflight_workers(self):
        cfg=config();remaining=64*1024**2
        runner=campaign.Runner('.',cfg,SimpleNamespace(remaining_bytes=remaining))
        for training,inflight in [(True,0),(False,0),(False,1)]:
            worker=cfg['storage']['fit_write_bytes'] if training else (inflight+1)*cfg['storage']['game_write_bytes']
            boundary=cfg['storage']['free_floor_bytes']+worker+remaining+campaign.FOREIGN_RECEIPT_RESERVE
            with patch.object(campaign,'owned_size',return_value=0):
                with patch.object(campaign.shutil,'disk_usage',return_value=SimpleNamespace(free=boundary-1)):
                    with self.assertRaises(AssertionError):runner.storage_ready(inflight,training)
                with patch.object(campaign.shutil,'disk_usage',return_value=SimpleNamespace(free=boundary)):
                    runner.storage_ready(inflight,training)

    def test_registered_full_panel_fits_before_every_dispatch_and_extra_refused(self):
        cfg=config();runner=campaign.Runner('.',cfg,None)
        for split,seeds in [('dev',cfg['validation_seeds']),('holdout',cfg['test_seeds'])]:
            for seed in seeds:
                for job in campaign.plans(cfg,split,seed):
                    runner.reserve(job);runner.attempts.append(job)
        self.assertEqual(len(runner.attempts),120)
        self.assertEqual(sum(j['policy']=='teacher_task' for j in runner.attempts),40)
        with self.assertRaises(AssertionError):runner.reserve(runner.attempts[0])
        with self.assertRaises(AssertionError):campaign.plans(cfg,'dev',201)

    def test_failed_offline_gate_dispatches_no_games_and_preserves_normal_stop(self):
        with _scratch() as folder:
            out=folder/'output';out.mkdir()
            campaign.write_new(out/'registration.json',{'synthetic':True})
            campaign.write_new(out/'freeze.json',{'synthetic':True})
            with patch.object(campaign,'ready',return_value=config()), \
                 patch.object(campaign,'verify'), \
                 patch.object(campaign.Runner,'gpu_fit',return_value=False), \
                 patch.object(campaign.Runner,'execute',side_effect=AssertionError('Forbidden game dispatch')):
                campaign.run(out)
            result=campaign.read(out/'run-summary.json')
            self.assertEqual(result['status'],'stopped_offline_gate')
            self.assertEqual(result['completed_games'],0)
            self.assertFalse(result['holdout_run'])
            self.assertTrue(campaign.read(out/'campaign-storage.json')['passed'])
            self.assertFalse((out/'failure.json').exists())
        self.assertFalse(folder.exists())

    def test_failure_before_runner_is_not_a_passed_storage_receipt(self):
        original=campaign.write_new
        def injected(path,value):
            if path.name=='campaign-started.json':raise OSError('Synthetic write failure')
            return original(path,value)
        with _scratch() as folder:
            out=folder/'output';out.mkdir()
            original(out/'registration.json',{'synthetic':True})
            original(out/'freeze.json',{'synthetic':True})
            with patch.object(campaign,'ready',return_value=config()), \
                 patch.object(campaign,'write_new',side_effect=injected):
                with self.assertRaisesRegex(OSError,'Synthetic'):campaign.run(out)
            self.assertFalse(campaign.read(out/'campaign-storage.json')['passed'])
            failure=campaign.read(out/'failure.json')
            self.assertEqual((failure['attempted_games'],failure['completed_games']),(0,0))
            self.assertFalse(failure['retry_allowed'])
        self.assertFalse(folder.exists())


if __name__=='__main__':unittest.main()
