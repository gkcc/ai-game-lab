import unittest
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch
from . import experiment as e

class ExperimentTests(unittest.TestCase):
    def test_private_checkpoint_creates_directory_and_preserves_existing_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; stage=root/'stage'; stage.mkdir(parents=True)
            outside=Path(tmp)/'outside'; outside.mkdir()
            with patch.object(e,'ROOT',root),patch.object(e,'OUT',stage):
                with self.assertRaises(FileNotFoundError): e.prepare_private_checkpoint(stage)
                self.assertFalse((stage/'private').exists())
                path=e.prepare_private_checkpoint(stage,create=True)
                self.assertEqual(path,stage/'private/model-final.npz')
                self.assertTrue(path.parent.is_dir())
                with path.open('xb') as stream: stream.write(b'prior checkpoint')
                with self.assertRaises(FileExistsError): e.prepare_private_checkpoint(stage)
                self.assertEqual(path.read_bytes(),b'prior checkpoint')
                with self.assertRaises(ValueError): e.prepare_private_checkpoint(outside)

    def test_private_checkpoint_rejects_external_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; stage=root/'stage'; stage.mkdir(parents=True)
            outside=Path(tmp)/'outside'; outside.mkdir(); sentinel=outside/'sentinel'; sentinel.write_bytes(b'unchanged')
            link=stage/'private'
            try:
                link.symlink_to(outside,target_is_directory=True)
            except (OSError,NotImplementedError) as exc:
                if os.name!='nt': self.skipTest('Creating a symbolic link is unavailable: '+type(exc).__name__)
                try:
                    subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(link),str(outside)],
                                   check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                except (OSError,subprocess.CalledProcessError) as junction_error:
                    self.skipTest('Creating a directory link is unavailable: '+type(junction_error).__name__)
            try:
                self.assertTrue(link.is_symlink() or getattr(link,'is_junction',lambda:False)())
                self.assertEqual(link.resolve(strict=True),outside.resolve(strict=True))
                with patch.object(e,'ROOT',root),patch.object(e,'OUT',stage):
                    with self.assertRaises(ValueError): e.prepare_private_checkpoint(stage)
                self.assertEqual(sentinel.read_bytes(),b'unchanged')
                self.assertFalse((outside/'model-final.npz').exists())
            finally:
                link.rmdir() if os.name=='nt' else link.unlink()

    def test_seed_overlap_and_types_rejected(self):
        e.validate_roles([1,2],[3,4],[5,6])
        for roles in (([1,2],[2,3],[4]),([1],[2],[1]),([True],[2],[3])):
            with self.assertRaises(ValueError): e.validate_roles(*roles)
    def test_role_scanner_does_not_exclude_all_historical_seeds(self):
        value={'excluded_uint32':[1,2],'train_seeds':[3], 'reserved_test_seeds':[4,5], 'holdout':{'seeds':[6]},'validation_seeds':[7]}
        self.assertEqual(e.role_values(value),{4,5,6,7})
    def test_seed_count_is_not_a_seed(self):
        registration={'holdout_seed_count':50374,'reserved_seeds_count':42,
                      'reserved_test_seeds':[11,12], 'validation_seeds':[13],
                      'holdout_seeds':[14], 'test_seeds':[15]}
        self.assertEqual(e.role_values(registration),{11,12,13,14,15})
        self.assertEqual(e.role_values(registration,development=True),{11,12,14,15})
    def test_validation_role_depends_on_bound_source(self):
        data={'experiment':'task-action-observation-v1','data_role':'Fresh development observation only; no promotion, tuning, confidence interval or independent holdout claim','validation_seeds':[1,2],'test_seeds':[3]}
        import json
        raw=json.dumps(data).encode()
        with patch.object(e,'git',return_value=raw),patch.object(e,'read',return_value={'source_root':'artifacts/task-action-observation-v1-20261027','train_seeds':[1,2]}):
            label='local/task-action-observation-v1-20261027/registration.json'
            self.assertTrue(e.development_record(label,raw))
            self.assertEqual(e.role_values(data,development=e.development_record(label,raw)),{3})
            self.assertFalse(e.development_record('local/other/registration.json',raw))
            self.assertEqual(e.role_values(data),{1,2,3})
            with patch.object(e,'git',return_value=b'changed'):
                with self.assertRaises(ValueError): e.development_record(label,raw)
    def test_build_config_requires_exact_roles_and_seed(self):
        oc={'train_seeds':[1,2],'training':{'training_seed':e.TRAINING['training_seed']},'schema_sha256':'x','wsl_software_versions':{},'engine_commit':'x','engine_sha256':'x','official_loader_sha256':'x'}
        nc={'train_seeds':[3,4]}; old={'selected_operations':{'DIG':2},'groups':2,'candidate_rows':2}
        new={'coverage':{'groups':2,'retained_selected_operations':{'DIG':2}},'sample_summary':{'groups':2,'selected_operations':{'DIG':2}},'full_stream':{'retained_candidate_rows':2}}
        inv={'holdout_seeds':[5],'scope':'synthetic','limitation':'synthetic'}
        cfg={'old_train_seeds':[1,2],'new_train_seeds':[3,4],'holdout_seed_count':1,'training_seed':e.TRAINING['training_seed'],'deadline_utc':'x'}
        args=(oc,nc,old,new,{'old':e.ROOT/'a','new':e.ROOT/'b'},{'old':'a','new':'b'},inv,{},'launch',{}, {})
        self.assertEqual(e.build_config(cfg,*args)['expected_groups'],{'old':2,'new':2})
        self.assertNotIn('holdout_seeds',e.build_config(cfg,*args))
        cfg['holdout_seed_count']=2
        with self.assertRaises(ValueError): e.build_config(cfg,*args)
    def test_runtime_line_endings_are_narrowly_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rel='decision/execution.py'; path=root/rel; path.parent.mkdir()
            local=b'a\r\nb\r\n'; path.write_bytes(local)
            freeze={'runtime_files':{rel:hashlib.sha256(local).hexdigest()}}
            with patch.object(e,'ROOT',root),patch.object(e,'REUSED',()),patch.object(e,'read',return_value=freeze),patch.object(e,'git',return_value=b'a\nb\n'):
                local_hash,public_hash=e.runtime_files()
                self.assertNotEqual(local_hash[rel],public_hash[rel])
                path.write_bytes(b'a\r\nchanged\r\n')
                with self.assertRaises(ValueError): e.runtime_files()
            path.write_bytes(local)
            with patch.object(e,'ROOT',root),patch.object(e,'REUSED',()),patch.object(e,'read',return_value=freeze),patch.object(e,'git',return_value=b'a\nchanged\n'):
                with self.assertRaises(ValueError): e.runtime_files()
    def test_job_cleanup_required(self):
        job=dict(passed=True,assigned=True,resumed=True,launcher_reaped=True,job_members_zero=True,exit_code=0,error_kind=None,cleanup_error_kind=None,stop_reason=None)
        self.assertTrue(e.good_job(job)); job['job_members_zero']=False; self.assertFalse(e.good_job(job))
if __name__=='__main__': unittest.main()
