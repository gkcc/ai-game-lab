import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from learning.task_teacher_choice_bc_v1 import numeric,training
from learning.task_choice_v1.features import OPS
from . import experiment as real_experiment

exp=types.ModuleType('learning.task_teacher_merge_bc_v1.experiment')
exp.OUT=exp.ROOT=Path(__file__).resolve().parent/'synthetic'
for name in ('read','sha','ready','deadline_check','prepare_private_checkpoint','write_new'): setattr(exp,name,lambda *a:None)
spec=importlib.util.spec_from_file_location('learning.task_teacher_merge_bc_v1._entry_test',Path(__file__).with_name('gpu_entry.py'))
entry=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'learning.task_teacher_merge_bc_v1.experiment':exp}): spec.loader.exec_module(entry)


def array(seed,value):
    return dict(shared=np.full((1,275),value,np.float32),candidates=np.zeros((1,98),np.float32),
        offsets=np.array([0,1],np.int64),teacher_choice_label=np.array([0],np.int64),
        ops=np.array([0],np.uint8),seed=np.array([seed],np.uint32))


class WorkerTests(unittest.TestCase):
    def run_fake(self,change=None,launch_override=None,existing=False,missing=False):
        cfg=dict(wsl_software_versions={},archive_path={'old':'old.npz','new':'new.npz'},
            archive_sha256={'old':'old','new':'new'},schema_path={'old':'old.json','new':'new.json'},schema_sha256='schema',
            gpu_launch_config_sha256='launch',old_train_seeds=[1],new_train_seeds=[2],seed_inventory_sha256='inventory',holdout_seed_count=1,
            expected_groups={'old':1,'new':1},expected_candidates={'old':1,'new':1},
            expected_operations={name:{op:int(i==0) for i,op in enumerate(OPS)} for name in ('old','new')},
            data_limits=dict(max_groups=5,max_candidates=5,max_archive_bytes=100,max_uncompressed_bytes=1000),
            training=dict(training_seed=8))
        if change: cfg.update(change)
        marker=dict(attempt=1,retry_allowed=False,freeze_sha256='freeze',archive_sha256=cfg['archive_sha256'],host_token_sha256=entry.token_sha('secret'))
        guard=dict(attempt=1,retry_allowed=False,freeze_sha256='freeze',fit_marker_sha256='marker',host_token_sha256=marker['host_token_sha256'])
        def read(path): return {'fit-started.json':marker,'gpu-guard-started.json':guard,
            'gpu-launch.local.json':launch_override if launch_override is not None else dict(expected_device='synthetic-device',expected_device_kind='synthetic-kind'), 'old.json':{},
            'seed-inventory.local.json':{'holdout_seeds':[2] if change and 'holdout' in change else [3]}}[Path(path).name]
        def sha(path): return {'freeze.json':'freeze','registration.json':'registration','pre-execution-commit.json':'commit',
            'fit-started.json':'marker','gpu-guard-started.json':'guard','gpu-launch.local.json':'launch',
            'old.json':'schema','new.json':'schema','seed-inventory.local.json':'inventory','model-final.npz':'model'}[Path(path).name]
        parity=dict(groups=16,synchronized=True,requested_matmul_precision='highest',hardware_precision_measured=False)
        trained=dict(train_seeds={1,2},sha256='model',training_summary=dict(synchronized_updates=128,gpu_verified=True,
            weights_changed=True,nonzero_gradient_updates=1,inference_parity=parity))
        fake_os=types.SimpleNamespace(name='posix',environ={entry.TOKEN_ENV:'secret'},getsid=lambda _:1,getpid=lambda:1)
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp)/'project'; stage=project/'stage'; stage.mkdir(parents=True)
            checkpoint=stage/'private/model-final.npz'
            if existing:
                checkpoint.parent.mkdir(); checkpoint.write_bytes(b'existing checkpoint')
            def small_fit(groups,schema,path,*args):
                with path.open('xb') as stream: stream.write(b'synthetic checkpoint')
                return trained
            with patch.object(entry,'ROOT',project),patch.object(entry,'OUT',stage), \
                 patch.object(real_experiment,'ROOT',project),patch.object(real_experiment,'OUT',stage), \
                 patch.object(entry,'prepare_private_checkpoint',real_experiment.prepare_private_checkpoint), \
                 patch.object(entry,'os',fake_os),patch.object(entry,'ready',return_value=cfg),patch.object(entry,'read',side_effect=read), \
                 patch.object(entry,'sha',side_effect=sha),patch.object(entry,'check_software_versions'),patch.object(entry,'deadline_check'), \
                 patch.object(entry,'write_new') as write,patch.object(numeric,'read',side_effect=[array(1,1),array(2,2)]) as decode, \
                 patch.object(training,'fit',side_effect=small_fit) as fit:
                if not existing and not missing: real_experiment.prepare_private_checkpoint(stage,create=True)
                if change or launch_override is not None or existing or missing:
                    error=FileExistsError if existing else FileNotFoundError if missing else ValueError
                    with self.assertRaises(error): entry._worker_impl(stage,lambda _:None)
                    fit.assert_not_called(); write.assert_not_called()
                    if existing: self.assertEqual(checkpoint.read_bytes(),b'existing checkpoint')
                    if missing: self.assertFalse(checkpoint.parent.exists())
                else:
                    entry._worker_impl(stage,lambda _:None)
                    groups=fit.call_args.args[0]
                    self.assertEqual([int(g['shared'][0]) for g in groups],[1,2])
                    self.assertEqual([g['seed'] for g in groups],[1,2])
                    self.assertEqual(checkpoint.read_bytes(),b'synthetic checkpoint')
                    self.assertEqual(write.call_args.args[1]['archive_snapshot_reads'],{'old':1,'new':1})
                self.assertEqual(decode.call_count,1 if change and 'expected_groups' in change else 2)

    def test_stable_order_and_one_read_each(self): self.run_fake()
    def test_holdout_or_counts_stop_before_fit(self):
        self.run_fake({'holdout':True})
        self.run_fake({'expected_groups':{'old':2,'new':1}})

    def test_existing_checkpoint_stops_before_fit(self): self.run_fake(existing=True)
    def test_missing_prepared_directory_stops_before_fit(self): self.run_fake(missing=True)

    def test_empty_or_nonstr_device_identity_stops_before_fit(self):
        for launch in (dict(expected_device='',expected_device_kind='kind'),
                       dict(expected_device='device',expected_device_kind='  '),
                       dict(expected_device=None,expected_device_kind='kind')):
            with self.subTest(launch=launch): self.run_fake(launch_override=launch)


if __name__=='__main__': unittest.main()
