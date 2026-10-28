import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import experiment as e
from .evaluation import GATES


class PrivateSeedTests(unittest.TestCase):
    def setUp(self):
        self.document = dict(version=1,seeds=[1111111111,2222222222,3333333333,4000000001],salt='ab'*32)

    def test_domain_separated_commitment_and_invalid_roles(self):
        expected=hashlib.sha256(e.SEED_DOMAIN+bytes.fromhex(self.document['salt'])+
            struct.pack('>4I',*self.document['seeds'])).hexdigest()
        self.assertEqual(e.seed_commitment(self.document),expected)
        for document in ({**self.document,'seeds':[1,1,2,3]}, {**self.document,'salt':'z'*64},
                         {**self.document,'seeds':[True,2,3,4]}, {**self.document,'version':True}):
            with self.assertRaises(ValueError): e.seed_commitment(document)

    def test_public_metadata_does_not_contain_private_values(self):
        e.ensure_redacted({'seed_count':4,'commitment':e.seed_commitment(self.document)},
                          self.document['seeds'],self.document['salt'])
        with self.assertRaises(ValueError):
            e.ensure_redacted({'nested':[{'other':self.document['seeds'][0]}]},self.document['seeds'],self.document['salt'])
        with self.assertRaises(ValueError):
            e.ensure_redacted({'other':self.document['salt']},self.document['seeds'],self.document['salt'])

    def test_seed_inventory_ignores_count_and_outcome_numbers(self):
        record={'seed_count':10,'seeds_count':11,'holdout_seed_count':99,'training_seeds':[1,2], 'test_seeds':[3],
                'nested':{'reserved_seeds':[4],'profit':5},'score':6,
                'excluded_uint32':[7], 'validation_seed':8}
        self.assertEqual(e._seed_values(record),{1,2,3,4,7,8})

    def test_private_snapshot_commitment_and_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; out=root/'artifacts'/e.STAGE_NAME; out.mkdir(parents=True)
            doc=out/e.PRIVATE_SEEDS; inv=out/'seed-inventory.local.json'
            doc.write_text(json.dumps(self.document),encoding='utf-8')
            inv.write_text(json.dumps({'excluded_uint32':[5,6]}),encoding='utf-8')
            cfg=dict(seed_count=4,seed_commitment=e.seed_commitment(self.document),
                     seed_document_sha256=e.sha(doc),seed_inventory_sha256=e.sha(inv))
            with patch.object(e,'ROOT',root),patch.object(e,'OUT',out):
                self.assertEqual(e.load_seeds(out,cfg),self.document['seeds'])
                with self.assertRaises(ValueError): e.load_seeds(out,{**cfg,'seed_commitment':'0'*64})
                inv.write_text(json.dumps({'excluded_uint32':[self.document['seeds'][0]]}),encoding='utf-8')
                with self.assertRaises(ValueError): e.load_seeds(out,{**cfg,'seed_inventory_sha256':e.sha(inv)})
                doc.write_text(json.dumps({**self.document,'salt':'cd'*32}),encoding='utf-8')
                with self.assertRaises(ValueError): e.load_seeds(out,cfg)

    def test_duplicate_private_document_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'private.json'
            path.write_text('{"version":1,"version":1,"seeds":[],"salt":"x"}',encoding='utf-8')
            with self.assertRaises(ValueError): e._bounded_json(path,e.sha(path),16384)

    def test_predecision_inventory_uses_explicit_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); artifacts=root/'artifacts'; artifact=artifacts/'existing'; artifact.mkdir(parents=True)
            (artifact/'registration.json').write_text(json.dumps({'test_seeds':[17]}),encoding='utf-8')
            calls=[]
            with patch.object(e,'ROOT',root),patch.object(e,'OUT',artifacts/'new'),patch.object(e,'git',return_value=b''):
                result=e.inventory(check=lambda:calls.append(True))
            self.assertEqual(result['excluded_uint32'],[17])
            self.assertEqual(len(calls),1)

    def test_historical_receipt_chain_rejects_changed_evidence(self):
        names=('registration.json','freeze.json','pre-review.json','privacy-revision.json','model-seal.json',
               'fit-result.json','fit-transport-result.json','gpu-guard-result.json','independent-result-assessment.json',
               'old-model-seal.json','old-registration.json')
        h={name:'sha:'+name for name in names}
        parent=dict(passed=True,competitive_gain_proven=False,automatic_promotion=False,checkpoint_sha256='model',
            registration_sha256=h['registration.json'],freeze_sha256=h['freeze.json'],pre_review_sha256=h['pre-review.json'],
            privacy_revision_sha256=h['privacy-revision.json'],model_seal_sha256=h['model-seal.json'],
            fit_result_sha256=h['fit-result.json'],fit_transport_sha256=h['fit-transport-result.json'],
            gpu_guard_sha256=h['gpu-guard-result.json'],independent_assessment_sha256=h['independent-result-assessment.json'],
            pre_execution_commit='commit')
        seal=dict(passed=True,checkpoint_sha256='model',freeze_sha256=h['freeze.json'],fit_result_sha256=h['fit-result.json'],
                  fit_transport_sha256=h['fit-transport-result.json'],linux_guard_sha256=h['gpu-guard-result.json'])
        fit=dict(passed=True,checkpoint_sha256='model',freeze_sha256=h['freeze.json'])
        transport=dict(passed=True); guard=dict(passed=True,freeze_sha256=h['freeze.json'])
        old=dict(passed=True); acceptance=dict(passed=True,evidence_sha256={'model-seal.json':h['old-model-seal.json'],
                                                                           'registration.json':h['old-registration.json']})
        args=(parent,seal,fit,transport,guard,old,acceptance,h,'commit')
        e.validate_parent_evidence(*args)
        with self.assertRaises(ValueError): e.validate_parent_evidence({**parent,'fit_transport_sha256':'changed'},*args[1:])
        with self.assertRaises(ValueError): e.validate_parent_evidence(parent,seal,{**fit,'checkpoint_sha256':'changed'},*args[3:])
        with self.assertRaises(ValueError): e.validate_parent_evidence(parent,seal,fit,transport,{**guard,'freeze_sha256':'changed'},*args[5:])

    def test_execution_commit_cannot_change_historical_runtime_or_evidence(self):
        freeze={'new_source_sha256':{},'public_runtime_sha256':{'learning/old.py':'sha'},
                'historical_evidence_sha256':{'old/registration.json':'sha'}}
        def fake_read(path):
            return {'commit':'candidate'} if Path(path).name=='pre-execution-commit.json' else freeze
        def git_for(changed):
            def git(*args):
                if args[0]=='show': return b'frozen bytes'
                if args[0]=='diff': return changed.encode()+b'\0'
                return b''
            return git
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; stage=root/'artifacts'/e.STAGE_NAME; stage.mkdir(parents=True)
            with patch.object(e,'ROOT',root),patch.object(e,'OUT',stage), \
                 patch.object(e,'verify',return_value={'ready':True}),patch.object(e,'read',side_effect=fake_read), \
                 patch.object(e,'clean_head'),patch.object(Path,'read_bytes',return_value=b'frozen bytes'):
                for changed in ('kaggriculture/learning/old.py','review/evidence/old/registration.json'):
                    with patch.object(e,'git',side_effect=git_for(changed)),self.assertRaises(ValueError):
                        e.verify_commit(stage)
                with patch.object(e,'git',side_effect=git_for('review/PROGRESS.md')):
                    self.assertEqual(e.verify_commit(stage),{'ready':True})

    def test_stage_storage_budget_is_fixed_and_covers_registered_panel(self):
        self.assertEqual(e.STORAGE['total_bytes'],10*e.GIB)
        self.assertEqual(e.STORAGE['game_write_bytes'],128*e.MIB)
        self.assertEqual(e.STORAGE['free_floor_bytes'],4*e.GIB)
        per_game=e.STORAGE['game_write_bytes']+e.STORAGE['process_log_bytes']+4*e.MIB
        self.assertLess(e.OFFICIAL['new_games']*per_game,e.STORAGE['total_bytes'])

        class ReachedHistory(Exception): pass
        cfg=dict(baseline_head=e.BASE,seed_count=4,seed_commitment='commitment',seed_document_sha256='docsha',
            opponents=['round2','market','teacher'],seats=[0,1],policies=['merged','unweighted','teacher_task'],
            storage=dict(e.STORAGE),official_job_supervision=e.JOB,independent_job_supervision=e.JOB,
            host_preflight=e.PREFLIGHT,independent_limits=e.INDEPENDENT,official_limits=e.OFFICIAL,gates=GATES,
            no_retraining=True,no_automatic_promotion=True,retry_allowed=False)
        decision=dict(seed_commitment='commitment',seed_document_sha256='docsha')
        freeze=dict(parent_commit=e.BASE,registration_sha256='sha',root_decision_sha256='sha',historical_evidence_sha256={})
        def fake_read(path):
            return {'registration.json':cfg,'root-decision.json':decision,'freeze.json':freeze}[Path(path).name]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; stage=root/'artifacts'/e.STAGE_NAME; stage.mkdir(parents=True)
            with patch.object(e,'ROOT',root),patch.object(e,'OUT',stage),patch.object(e,'deadline_check'),patch.object(e,'read',side_effect=fake_read), \
                 patch.object(e,'sha',return_value='sha'),patch.object(e,'historical_inputs',side_effect=ReachedHistory):
                with self.assertRaises(ReachedHistory): e.verify(stage)
                cfg['storage']={**e.STORAGE,'total_bytes':4*e.GIB}
                with self.assertRaisesRegex(ValueError,'Frozen panel or budget changed'): e.verify(stage)

    def _link(self,path,target):
        if os.name=='nt':
            try:
                subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(path),str(target)],
                               check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            except (OSError,subprocess.CalledProcessError) as error:
                self.skipTest('Directory junction unavailable: '+type(error).__name__)
            is_junction=getattr(path,'is_junction',None)
            if is_junction is not None: self.assertTrue(is_junction())
        else:
            try:
                path.symlink_to(target,target_is_directory=True)
            except (OSError,NotImplementedError) as exc:
                self.skipTest('Directory link unavailable: '+type(exc).__name__)

    def _unlink(self,path):
        path.rmdir() if os.name=='nt' else path.unlink()

    def test_output_stage_link_is_rejected_before_writing_or_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; artifacts=root/'artifacts'; artifacts.mkdir(parents=True)
            outside=Path(tmp)/'outside'; outside.mkdir(); sentinel=outside/'sentinel'; sentinel.write_bytes(b'keep')
            stage=artifacts/e.STAGE_NAME; self._link(stage,outside)
            try:
                with patch.object(e,'ROOT',root),patch.object(e,'OUT',stage):
                    for action in (lambda:e.write_new(stage/'new.json',{'x':1}),lambda:e.load_seeds(stage,{}),
                                   lambda:e.prepare(stage),lambda:e.verify(stage),lambda:e.verify_commit(stage),lambda:e.ready(stage)):
                        with self.assertRaises(ValueError): action()
                    with self.assertRaises(FileExistsError): e.create_out()
                self.assertEqual(sentinel.read_bytes(),b'keep')
                self.assertFalse((outside/'new.json').exists())
            finally: self._unlink(stage)

    def test_artifacts_ancestor_link_and_nested_link_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; root.mkdir(); outside=Path(tmp)/'outside'; outside.mkdir()
            sentinel=outside/'sentinel'; sentinel.write_bytes(b'keep')
            artifacts=root/'artifacts'; self._link(artifacts,outside); stage=artifacts/e.STAGE_NAME
            try:
                with patch.object(e,'ROOT',root),patch.object(e,'OUT',stage):
                    with self.assertRaises(ValueError): e.write_new(stage/'new.json',{'x':1})
                    with self.assertRaises(ValueError): e.create_out()
                self.assertEqual(sentinel.read_bytes(),b'keep')
                self.assertFalse((outside/e.STAGE_NAME/'new.json').exists())
            finally: self._unlink(artifacts)
            stage.mkdir(parents=True); nested=stage/'nested'; self._link(nested,outside)
            try:
                with patch.object(e,'ROOT',root),patch.object(e,'OUT',stage):
                    with self.assertRaises(ValueError): e.write_new(nested/'new.json',{'x':1})
                self.assertEqual(sentinel.read_bytes(),b'keep')
                self.assertFalse((outside/'new.json').exists())
            finally: self._unlink(nested)

    def test_stage_creation_requires_trusted_parent_and_never_adopts_existing_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; artifacts=root/'artifacts'; artifacts.mkdir(parents=True)
            stage=artifacts/e.STAGE_NAME
            with patch.object(e,'ROOT',root),patch.object(e,'OUT',stage):
                self.assertEqual(e.create_out(),stage)
                sentinel=stage/'sentinel'; sentinel.write_bytes(b'keep')
                with self.assertRaises(FileExistsError): e.create_out()
                self.assertEqual(sentinel.read_bytes(),b'keep')

    def test_regular_stage_is_confined_and_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; stage=root/'artifacts'/e.STAGE_NAME; stage.mkdir(parents=True)
            target=stage/'owned'/'record.json'
            with patch.object(e,'ROOT',root),patch.object(e,'OUT',stage):
                e.write_new(target,{'passed':True})
                original=target.read_bytes()
                with self.assertRaises(FileExistsError): e.write_new(target,{'passed':False})
                with self.assertRaises(ValueError): e.write_new(stage/'..'/'outside.json',{'x':1})
                self.assertEqual(target.read_bytes(),original)
                self.assertFalse((root/'artifacts'/'outside.json').exists())


if __name__=='__main__': unittest.main()
