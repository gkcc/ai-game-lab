import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy
from . import experiment as real_experiment

exp=types.ModuleType('learning.task_teacher_merge_bc_v1.experiment')
exp.OUT=exp.ROOT=Path(__file__).resolve().parent/'synthetic'
for name in ('deadline_check','prepare_private_checkpoint','read','ready','sha','verify_commit','write_new'): setattr(exp,name,lambda *a:None)
spec=importlib.util.spec_from_file_location('learning.task_teacher_merge_bc_v1._campaign_test',Path(__file__).with_name('campaign.py'))
c=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_teacher_merge_bc_v1.experiment':exp}): spec.loader.exec_module(c)


class CampaignTests(unittest.TestCase):
    def test_existing_private_content_stops_before_marker_and_spawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp)/'project'; stage=project/'stage'; private=stage/'private'; private.mkdir(parents=True)
            saved=private/'unrelated.bin'; saved.write_bytes(b'preserve')
            cfg={'version':1}
            with patch.object(c,'OUT',stage),patch.object(real_experiment,'ROOT',project),patch.object(real_experiment,'OUT',stage), \
                 patch.object(c,'prepare_private_checkpoint',real_experiment.prepare_private_checkpoint), \
                 patch.object(c,'ready',return_value=cfg),patch.object(c,'verify_commit'), \
                 patch.object(c,'memory_ready',return_value=True),patch.object(c,'marker') as marker, \
                 patch.object(c,'CapturedProcess') as spawn:
                with self.assertRaises(FileExistsError): c.fit()
                marker.assert_not_called(); spawn.assert_not_called()
            self.assertEqual(saved.read_bytes(),b'preserve')

    def test_private_output_error_stops_before_marker_and_spawn(self):
        cfg={'version':1}
        with patch.object(c,'ready',return_value=cfg),patch.object(c,'verify_commit'), \
             patch.object(Path,'exists',return_value=False),patch.object(c,'memory_ready',return_value=True), \
             patch.object(c,'prepare_private_checkpoint',side_effect=FileExistsError('existing content')), \
             patch.object(c,'marker') as marker,patch.object(c,'CapturedProcess') as spawn:
            with self.assertRaises(FileExistsError): c.fit()
            marker.assert_not_called(); spawn.assert_not_called()

    def test_preflight_config_drift_stops_before_marker_and_spawn(self):
        with patch.object(c,'ready',side_effect=[{'version':1},{'version':2}]),patch.object(c,'verify_commit'), \
             patch.object(Path,'exists',return_value=False),patch.object(c,'memory_ready',return_value=True), \
             patch.object(c,'marker') as marker,patch.object(c,'CapturedProcess') as spawn:
            with self.assertRaises(ValueError): c.fit()
            marker.assert_not_called(); spawn.assert_not_called()

    def test_fit_acceptance_requires_both_snapshots_and_parity(self):
        cfg=dict(archive_sha256={'old':'old','new':'new'},schema_sha256='schema',seed_inventory_sha256='inventory',expected_groups={'old':1,'new':2})
        identities={'freeze.json':'freeze','registration.json':'registration','pre-execution-commit.json':'commit',
                    'fit-started.json':'marker','gpu-launch.local.json':'launch'}
        fitted=dict(passed=True,freeze_sha256='freeze',archive_sha256=cfg['archive_sha256'],schema_sha256='schema',
            archive_snapshot_reads=dict(old=1,new=1),archive_read_scope=c.ARCHIVE_READ_SCOPE,groups=3,updates=128,nonzero_gradient_updates=1,weights_changed=True,
            gpu_verified=True,retry_allowed=False,checkpoint_sha256='checkpoint',identities=dict(freeze='freeze',
            registration='registration',pre_execution_commit='commit',marker='marker',guard_marker='guard',
            launch='launch',schema='schema',new_schema='schema',seed_inventory='inventory'),inference_parity=dict(groups=16,candidate_rows=16,
            synchronized=True,requested_matmul_precision='highest',hardware_precision_measured=False,max_abs_error=0.0))
        with patch.object(c,'sha',side_effect=lambda p:'guard' if Path(p).name=='gpu-guard-started.json' else 'checkpoint'):
            c.validate_fit(fitted,cfg,exp.OUT,'freeze',identities)
            for change in (dict(archive_snapshot_reads={'old':1,'new':0}),dict(archive_read_scope='one read'),dict(weights_changed=False),dict(updates=127)):
                with self.assertRaises(ValueError): c.validate_fit({**fitted,**change},cfg,exp.OUT,'freeze',identities)


if __name__=='__main__': unittest.main()
