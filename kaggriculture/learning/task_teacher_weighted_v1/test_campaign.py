import types
import unittest
from unittest.mock import patch
from . import campaign


class CampaignTest(unittest.TestCase):
    def test_preflight_does_not_proceed_after_low_sample(self):
        cfg=dict(host_limits=dict(preflight_samples=3,preflight_min_available_bytes=4,preflight_interval_seconds=10))
        with patch.object(campaign,'deadline_check'):
            self.assertFalse(campaign.memory_ready(cfg,available=iter([5,3,5]).__next__,sleep=lambda _:None))
            self.assertTrue(campaign.memory_ready(cfg,available=iter([4,5,6]).__next__,sleep=lambda _:None))

    def test_host_floor_cancels_owned_linux_session_and_does_not_seal(self):
        class Process:
            def __init__(self): self.done=False
            def poll(self): return 0 if self.done else None
            def wait(self,timeout): self.done=True; return 0
        child=types.SimpleNamespace(process=Process(),check=lambda:None)
        cfg=dict(host_limits=dict(min_available_bytes=2,poll_seconds=.25),process_log_bytes=100,
                 gpu_launch_config_sha256='a'*64,training=dict(process_seconds=900,cancellation_wait_seconds=20,cleanup_seconds=10),
                 cache_sha256='b'*64)
        launch=dict(distro='synthetic',workdir='synthetic',python='synthetic')
        saved=[]
        def mark(out,value,state): state.update(created=True,complete=True)
        clean=dict(owned_instances_remaining=0,reader_joined=True,launched_process_reaped=True)
        with patch.object(campaign,'ready',return_value=cfg),patch.object(campaign,'verify_commit'), \
            patch.object(campaign,'memory_ready',return_value=True),patch.object(campaign,'deadline_check'), \
            patch.object(campaign.Path,'exists',return_value=False),patch.object(campaign,'sha',return_value='a'*64), \
            patch.object(campaign,'marker',side_effect=mark),patch.object(campaign,'read',return_value=launch), \
            patch.object(campaign,'write_new',side_effect=lambda p,v:saved.append((p.name,v))), \
            patch.object(campaign,'CapturedProcess',return_value=child),patch.object(campaign,'observe_processes'), \
            patch.object(campaign,'reap',return_value=clean), \
            patch.object(campaign.psutil,'virtual_memory',return_value=types.SimpleNamespace(available=1)):
            with self.assertRaisesRegex(RuntimeError,'Fit or cleanup failed'): campaign.fit()
        receipts=dict(saved)
        self.assertIn('gpu-cancel.local.json',receipts)
        self.assertEqual(receipts['fit-transport-result.json']['stop_reason'],'available_memory_limit')
        self.assertFalse(receipts['fit-transport-result.json']['linux_cleanup_proven'])
        self.assertNotIn('model-seal.json',receipts)

    def test_fit_semantics_require_snapshot_weights_and_parity(self):
        cfg=dict(expected_class_weights={'DIG':2.},expected_operations={'DIG':8},cache_sha256='a',schema_sha256='b')
        identities={'freeze.json':'f','registration.json':'r','pre-execution-commit.json':'p',
                    'fit-started.json':'m','gpu-launch.local.json':'l'}
        expected=dict(freeze='f',registration='r',pre_execution_commit='p',marker='m',guard_marker='checkpoint',launch='l',schema='b')
        parity=dict(groups=16,candidate_rows=16,synchronized=True,requested_matmul_precision='highest',
                    hardware_precision_measured=False,max_abs_error=1e-6)
        result=dict(passed=True,class_weights={'DIG':2.},freeze_sha256='f',cache_sha256='a',schema_sha256='b',
            operations={'DIG':8},archive_snapshot_reads=1,updates=128,nonzero_gradient_updates=128,weights_changed=True,
            gpu_verified=True,retry_allowed=False,inference_parity=parity,checkpoint_sha256='checkpoint',identities=expected)
        with patch.object(campaign,'sha',return_value='checkpoint'):
            campaign.validate_fit(result,cfg,campaign.OUT,'f',identities)
            for bad in ({**result,'archive_snapshot_reads':True},{**result,'class_weights':{'DIG':1.}},
                        {**result,'inference_parity':{**parity,'candidate_rows':15}},
                        {**result,'identities':{**expected,'marker':'changed'}}):
                with self.subTest(bad=bad),self.assertRaises(ValueError): campaign.validate_fit(bad,cfg,campaign.OUT,'f',identities)

    def test_observation_error_still_invokes_reaper(self):
        class Process:
            def __init__(self): self.done=False
            def poll(self): return 0 if self.done else None
            def wait(self,timeout): self.done=True; return 0
        child=types.SimpleNamespace(process=Process(),check=lambda:None)
        cfg=dict(host_limits=dict(min_available_bytes=2,poll_seconds=.25),process_log_bytes=100,
                 gpu_launch_config_sha256='a'*64,training=dict(process_seconds=900,cancellation_wait_seconds=20,cleanup_seconds=10),
                 cache_sha256='b'*64)
        launch=dict(distro='synthetic',workdir='synthetic',python='synthetic'); saved=[]
        def mark(out,value,state): state.update(created=True,complete=True)
        clean=dict(owned_instances_remaining=0,reader_joined=True,launched_process_reaped=True)
        with patch.object(campaign,'ready',return_value=cfg),patch.object(campaign,'verify_commit'), \
            patch.object(campaign,'memory_ready',return_value=True),patch.object(campaign,'deadline_check'), \
            patch.object(campaign.Path,'exists',return_value=False),patch.object(campaign,'sha',return_value='a'*64), \
            patch.object(campaign,'marker',side_effect=mark),patch.object(campaign,'read',return_value=launch), \
            patch.object(campaign,'write_new',side_effect=lambda p,v:saved.append((p.name,v))), \
            patch.object(campaign,'CapturedProcess',return_value=child), \
            patch.object(campaign,'observe_processes',side_effect=PermissionError), \
            patch.object(campaign,'reap',return_value=clean) as reaper:
            with self.assertRaisesRegex(RuntimeError,'Fit or cleanup failed'): campaign.fit()
        reaper.assert_called_once()
        receipts=dict(saved)
        self.assertEqual(receipts['fit-transport-result.json']['observation_error_kind'],'PermissionError')
        self.assertFalse(receipts['fit-transport-result.json']['passed'])
        self.assertNotIn('model-seal.json',receipts)


if __name__=='__main__': unittest.main()
