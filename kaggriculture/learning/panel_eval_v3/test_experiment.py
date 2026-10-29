import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import experiment as e


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.root = e.ROOT
        self.paths = {name: self.root/'synthetic'/(name+'.npz') for name in ('merged','unweighted','crop')}
        self.hashes = {name: char*64 for name,char in (('merged','1'),('unweighted','2'),('crop','3'))}
        self.opponents = {name+'.py': self.root/'synthetic'/(name+'.py') for name in ('round2','market','teacher')}
        self.source_hashes = {name: char*64 for name,char in (('round2.py','4'),('market.py','5'),('teacher.py','6'))}
        self.parent = dict(engine_commit=e.ENGINE_COMMIT, engine_sha256=e.ENGINE_SHA256,
                           official_loader_sha256='a'*64)
        self.env = {'interpreter_sha256':'b'*64, 'versions':{'python':'synthetic'}}
        self.previous = dict(environment=copy.deepcopy(self.env), game_limits={'controller_calls':719})
        self.reference = dict(runtime_files={'learning/old.py':'c'*64},
            public_runtime_sha256={'learning/old.py':'d'*64},
            new_source_sha256={'learning/previous.py':'e'*64},
            historical_evidence_sha256={'old/registration.json':'f'*64})
        self.freeze = copy.deepcopy(self.reference)
        self.freeze['run_kind'] = e.FORMAL
        self.decision = dict(status='approved_for_implementation_not_execution',
            deadline_utc='2099-01-01T00:00:00+00:00', memory_wait=copy.deepcopy(e.MEMORY_WAIT),
            outer_supervision=copy.deepcopy(e.OUTER_SUPERVISION), scratch_helper_commitment='a'*64,run_kind=e.FORMAL)
        self.cfg = dict(seed_count=4, opponents=['round2','market','teacher'], seats=[0,1],
            policies=['merged','unweighted','teacher_task'], engine_commit=e.ENGINE_COMMIT,
            engine_sha256=e.ENGINE_SHA256, official_core_sha256=e.ENGINE_CORE_SHA256,
            official_loader_sha256=self.parent['official_loader_sha256'],
            environment=copy.deepcopy(self.env), game_limits=copy.deepcopy(self.previous['game_limits']),
            storage=copy.deepcopy(e.old.STORAGE), official_job_supervision=copy.deepcopy(e.old.JOB),
            independent_job_supervision=copy.deepcopy(e.old.JOB), host_preflight=copy.deepcopy(e.old.PREFLIGHT),
            outer_supervision=copy.deepcopy(e.OUTER_SUPERVISION),
            scratch_helper_commitment='a'*64,
            independent_limits=copy.deepcopy(e.old.INDEPENDENT), official_limits=copy.deepcopy(e.old.OFFICIAL),
            gates=copy.deepcopy(e.evaluation.GATES), no_retraining=True, no_automatic_promotion=True,
            retry_allowed=False, independent_selection=e.INDEPENDENT_SELECTION,
            deadline_utc=self.decision['deadline_utc'], source_sha256=copy.deepcopy(self.source_hashes))
        self.cfg.update(copy.deepcopy(e.run_contract(e.FORMAL)))
        for name,path in self.paths.items():
            self.cfg[name+'_checkpoint'] = path.relative_to(self.root).as_posix()
            self.cfg[name+'_checkpoint_sha256'] = self.hashes[name]

    def sha(self, path):
        return self.source_hashes[Path(path).name]

    def check(self):
        with patch.object(e.old, 'sha', side_effect=self.sha):
            return e._validate_contract(self.cfg, self.decision, self.freeze, self.reference,
                self.parent, self.paths, self.hashes, self.opponents, self.previous, self.env)

    def test_original_frozen_contract_is_accepted(self):
        self.assertEqual(self.check(), self.source_hashes)

    def test_engine_model_and_budget_changes_are_rejected(self):
        modifications = (
            ('engine_commit','wrong'), ('engine_sha256','0'*64), ('official_core_sha256','0'*64),
            ('official_loader_sha256','0'*64),
            ('seed_count',4.0), ('seats',[False,True]), ('retry_allowed',True), ('no_retraining',False),
            ('no_automatic_promotion',False), ('independent_selection','other'),
            ('merged_checkpoint','elsewhere'), ('unweighted_checkpoint_sha256','0'*64),
            ('game_limits',{'controller_calls':718}), ('storage',{}), ('official_job_supervision',{}),
            ('independent_job_supervision',{}), ('host_preflight',{}), ('outer_supervision',{}),
            ('scratch_helper_commitment','0'*64), ('independent_limits',{}),
            ('official_limits',{}), ('gates',{}), ('environment',{}))
        for key,value in modifications:
            with self.subTest(field=key):
                original=self.cfg[key]; self.cfg[key]=value
                with self.assertRaises(ValueError): self.check()
                self.cfg[key]=original

    def test_numeric_boolean_does_not_equal_registered_resource_limit(self):
        self.cfg['official_job_supervision']['wall_seconds'] = float(e.old.JOB['wall_seconds'])
        with self.assertRaises(ValueError): self.check()
        self.cfg['official_job_supervision'] = copy.deepcopy(e.old.JOB)
        self.cfg['host_preflight']['samples'] = True
        with self.assertRaises(ValueError): self.check()

    def test_wait_deadline_opponent_and_reference_changes_are_rejected(self):
        self.decision['memory_wait']['single_wait_seconds'] += 1
        with self.assertRaises(ValueError): self.check()
        self.decision['memory_wait']=copy.deepcopy(e.MEMORY_WAIT)
        self.decision['deadline_utc']='2098-01-01T00:00:00+00:00'
        with self.assertRaises(ValueError): self.check()
        self.decision['deadline_utc']=self.cfg['deadline_utc']
        self.cfg['source_sha256']['round2.py']='0'*64
        with self.assertRaises(ValueError): self.check()
        self.cfg['source_sha256']=copy.deepcopy(self.source_hashes)
        self.freeze['runtime_files']['learning/old.py']='0'*64
        with self.assertRaises(ValueError): self.check()

    def test_model_paths_cannot_escape_project(self):
        self.paths['crop']=self.root.parent/'outside.npz'
        with self.assertRaises(ValueError): self.check()

    def test_helper_commitment_requires_exact_review_binding(self):
        review = {'scratch_helper_commitment':self.cfg['scratch_helper_commitment'], 'passed':True,
                  'run_kind':e.FORMAL,
                  'review_basis':'independent_subagent', 'reviewer':'/root/synthetic',
                  'open_findings':[], 'checks':['source_identity']}
        e._validate_helper_review(self.cfg, review)
        for bad in ({}, {**review,'scratch_helper_commitment':'0'*64},
                    {**review,'scratch_helper_commitment':True}, {**review,'open_findings':['unresolved']}):
            with self.subTest(review=bad), self.assertRaises(ValueError):
                e._validate_helper_review(self.cfg, bad)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.reference=dict(runtime_files={'learning/old.py':'a'*64},
            public_runtime_sha256={'learning/old.py':'b'*64},
            new_source_sha256={'learning/prior.py':'c'*64},
            historical_evidence_sha256={'prior/registration.json':'d'*64})
        self.files={'learning/test-v2.py': b'synthetic source\n'}
        self.sha={name:hashlib.sha256(raw).hexdigest() for name,raw in self.files.items()}
        self.freeze={'parent_commit':'2'*40, 'supplemental_runtime_sha256':{name:dict(local_sha256=digest,public_sha256=digest)
                                                    for name,digest in self.sha.items()}}
        self.changed=b''
        self.parent_modified=False

    def read_regular(self,path,boundary):
        return self.files[Path(path).relative_to(e.ROOT).as_posix()]

    def sha_file(self,path):
        rel=Path(path).relative_to(e.ROOT).as_posix()
        return self.reference['runtime_files'].get(rel, self.reference['new_source_sha256'].get(rel))

    def git(self,*args):
        if args[0]=='show':
            if self.parent_modified and args[1].startswith('2'*40+':'): return b'different parent source'
            return self.files[args[1].split(':kaggriculture/',1)[1]]
        if args[0]=='diff': return self.changed
        raise AssertionError('Unexpected Git request')

    def check(self):
        with patch.object(e,'SUPPLEMENTAL_FILES',frozenset(self.files)), \
             patch.object(e.history,'_read_regular',side_effect=self.read_regular), \
             patch.object(e.old,'sha',side_effect=self.sha_file), patch.object(e.old,'git',side_effect=self.git):
            e._validate_sources(None,'1'*40,self.freeze,self.reference)

    def test_exact_supplemental_sources_and_prior_line_mapping(self):
        # Prior local/public hashes can differ only because the earlier freeze
        # already recorded that identity; supplemental files must match exactly.
        self.check()

    def test_missing_extra_or_modified_supplemental_source_is_rejected(self):
        name=next(iter(self.files)); entry=self.freeze['supplemental_runtime_sha256'].pop(name)
        with self.assertRaises(ValueError): self.check()
        self.freeze['supplemental_runtime_sha256'][name]=entry
        entry['public_sha256']='0'*64
        with self.assertRaises(ValueError): self.check()
        entry['public_sha256']=self.sha[name]
        self.freeze['supplemental_runtime_sha256']['unexpected.py']=copy.deepcopy(entry)
        with self.assertRaises(ValueError): self.check()

    def test_committed_history_change_is_rejected(self):
        self.changed=b'kaggriculture/learning/old.py\0'
        with self.assertRaises(ValueError): self.check()
        self.changed=b'review/evidence/prior/registration.json\0'
        with self.assertRaises(ValueError): self.check()

    def test_supplemental_source_must_exist_unchanged_before_registration(self):
        self.parent_modified=True
        with self.assertRaises(ValueError): self.check()


class ContextTests(unittest.TestCase):
    def setUp(self):
        # These contract fixtures are synthetic; separate guard tests exercise
        # the production fail-closed v2 consumed-seed requirement.
        prior=e.history.SeedInventory([901,902,903,904],0,0)
        guard=patch.object(e.prior_v2_guard,'require_v2_consumed_chain',return_value=prior)
        guard.start(); self.addCleanup(guard.stop)
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage=Path(self.temp.name)/'fresh-stage'; self.stage.mkdir()
        self.commit='a'*40
        self.patch=patch.object(e.old,'trusted_artifacts',return_value=Path(self.temp.name))
        self.patch.start(); self.addCleanup(self.patch.stop)

    def write(self,name,value):
        (self.stage/name).write_text(json.dumps(value),encoding='utf-8')

    def context(self):
        return e.Context(self.stage,self.commit)

    def test_deadline_rejects_expired_naive_or_mismatched_values(self):
        deadline=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        self.write('root-decision.json',{'deadline_utc':deadline})
        self.write('registration.json',{'deadline_utc':deadline})
        with self.assertRaises(TimeoutError): self.context().deadline_check()
        self.write('root-decision.json',{'deadline_utc':'2099-01-01T00:00:00'})
        self.write('registration.json',{'deadline_utc':'2099-01-01T00:00:00'})
        with self.assertRaises(ValueError): self.context().deadline_check()
        self.write('registration.json',{'deadline_utc':'other'})
        with self.assertRaises(ValueError): self.context().deadline_check()

    def test_exclusive_stage_output_and_traversal_rejection(self):
        context=self.context()
        target=self.stage/'results'/'receipt.json'
        context.write_new(target,{'passed':True})
        self.assertEqual(json.loads(target.read_text(encoding='utf-8')),{'passed':True})
        with self.assertRaises(FileExistsError): context.write_new(target,{'passed':False})
        with self.assertRaises(ValueError): context.write_new(self.stage/'..'/'outside.json',{})
        self.assertFalse((self.stage.parent/'outside.json').exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows path alias check')
    def test_windows_output_aliases_and_alternate_streams_are_rejected(self):
        context = self.context()
        for name in ('odd.', 'trailing ', 'CON', 'nul.json', 'base.json:hidden', 'sub/odd.'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                context.write_new(self.stage/name, {'passed':True})
        self.assertFalse((self.stage/'odd').exists())
        self.assertFalse((self.stage/'base.json').exists())

    def test_worker_context_exposes_trusted_artifact_and_reparse_checks(self):
        context=self.context()
        self.assertEqual(context.trusted_artifacts(),self.stage.parent)
        self.assertTrue(context._reject_reparse(self.stage))

    def test_ready_runs_rescan_before_other_identity_checks(self):
        deadline='2099-01-01T00:00:00+00:00'
        self.write('root-decision.json',{'deadline_utc':deadline})
        self.write('registration.json',{'deadline_utc':deadline})
        with patch.object(e.registered_rescan,'rescan_registered',side_effect=ValueError('untrusted stage')) as rescan, \
             patch.object(e.old,'historical_inputs') as inputs:
            with self.assertRaises(ValueError): self.context().ready()
            self.assertEqual(rescan.call_count,1)
            inputs.assert_not_called()

    def test_dynamic_rescan_must_include_all_prior_v2_roles(self):
        deadline='2099-01-01T00:00:00+00:00'
        self.write('root-decision.json',{'deadline_utc':deadline})
        self.write('registration.json',{'deadline_utc':deadline})
        missing=e.history.SeedInventory([901,902,903],0,0)
        with patch.object(e.registered_rescan,'rescan_registered',return_value=missing), \
             patch.object(e.old,'historical_inputs') as inputs:
            with self.assertRaises(ValueError): self.context().ready()
            inputs.assert_not_called()

    def test_ready_accepts_only_the_expected_engine_and_runs_rescan_twice(self):
        fixture=ContractTests('test_original_frozen_contract_is_accepted'); fixture.setUp()
        self.write('root-decision.json',fixture.decision)
        self.write('registration.json',fixture.cfg)
        self.write('freeze.json',fixture.freeze)
        self.write('pre-review.json',{'scratch_helper_commitment':fixture.cfg['scratch_helper_commitment'],
            'run_kind':e.FORMAL,
            'passed':True,'review_basis':'independent_subagent','reviewer':'/root/synthetic',
            'open_findings':[],'checks':['source_identity']})
        def sha(path):
            path=Path(path)
            return fixture.source_hashes[path.name]
        prior=e.history.SeedInventory([901,902,903,904],0,0)
        with patch.object(e.registered_rescan,'rescan_registered',return_value=prior) as rescan, \
             patch.object(e,'_reference_freeze',return_value=fixture.reference), \
             patch.object(e.old,'historical_inputs',return_value=(fixture.parent,fixture.paths,fixture.hashes,fixture.opponents)), \
             patch.object(e.old,'read',return_value=fixture.previous), patch.object(e.old,'environment',return_value=fixture.env), \
             patch.object(e.old,'sha',side_effect=sha), patch.object(e,'_check_official_runtime') as official, \
             patch.object(e,'_validate_sources') as sources:
            self.assertEqual(self.context().ready(),fixture.cfg)
            self.assertEqual(rescan.call_count,2)
            official.assert_called_once_with(fixture.parent['official_loader_sha256'])
            sources.assert_called_once()
        with patch.object(e.registered_rescan,'rescan_registered',return_value=prior), \
             patch.object(e,'_reference_freeze',return_value=fixture.reference), \
             patch.object(e.old,'historical_inputs',return_value=(fixture.parent,fixture.paths,fixture.hashes,fixture.opponents)), \
             patch.object(e.old,'read',return_value=fixture.previous), patch.object(e.old,'environment',return_value=fixture.env), \
             patch.object(e.old,'sha',side_effect=sha), patch.object(e,'_check_official_runtime',side_effect=ValueError('official changed')), \
             patch.object(e,'_validate_sources'):
            with self.assertRaises(ValueError): self.context().ready()

    def test_official_rules_loader_core_and_upstream_are_all_bound(self):
        rules=Path(self.temp.name)/'rules'; loader=Path(self.temp.name)/'loader'; core=Path(self.temp.name)/'core'
        files={'rules':rules,'loader':loader,'core':core}
        hashes={rules:e.ENGINE_SHA256, loader:'a'*64, core:e.ENGINE_CORE_SHA256}
        with patch.object(e,'_installed_files',return_value=files), \
             patch.object(e.old,'sha',side_effect=lambda p:hashes[Path(p)]), \
             patch.object(e.subprocess,'check_output',return_value=e.ENGINE_COMMIT.encode()) as head:
            e._check_official_runtime('a'*64)
            for path in (rules,loader,core):
                before=hashes[path]; hashes[path]='0'*64
                with self.assertRaises(ValueError): e._check_official_runtime('a'*64)
                hashes[path]=before
            head.return_value=b'0'*40
            with self.assertRaises(ValueError): e._check_official_runtime('a'*64)

    def test_seed_access_requires_matching_private_document_and_commitment(self):
        document={'version':1,'seeds':[101,202,303,404],'salt':'ab'*32}
        raw=json.dumps(document).encode('utf-8')
        (self.stage/'private-seeds.local.json').write_bytes(raw)
        cfg={'seed_document_sha256':hashlib.sha256(raw).hexdigest(),
             'seed_commitment':e.old.seed_commitment(document)}
        cfg.update(copy.deepcopy(e.run_contract(e.FORMAL)))
        self.write('registration.json',cfg)
        self.assertEqual(self.context().load_seeds(self.stage,cfg),document['seeds'])
        with self.assertRaises(ValueError): self.context().load_seeds(self.stage,{**cfg,'other':True})
        (self.stage/'private-seeds.local.json').write_bytes(raw+b' ')
        with self.assertRaises(ValueError): self.context().load_seeds(self.stage,cfg)

    def test_seed_access_rejects_reserved_v2_collision(self):
        document={'version':1,'seeds':[901,102,103,104],'salt':'ab'*32}
        raw=json.dumps(document).encode('utf-8')
        (self.stage/'private-seeds.local.json').write_bytes(raw)
        cfg={'seed_document_sha256':hashlib.sha256(raw).hexdigest(),
             'seed_commitment':e.old.seed_commitment(document)}
        cfg.update(copy.deepcopy(e.run_contract(e.FORMAL)))
        self.write('registration.json',cfg)
        with self.assertRaises(ValueError): self.context().load_seeds(self.stage,cfg)


if __name__ == '__main__':
    unittest.main()
