"""Synthetic registration tests; no real seeds or game execution."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import experiment as e
from . import prepare as p
from learning.panel_safety_v1.seed_inventory import SeedInventory
from learning.panel_safety_v2 import registered_rescan as r


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'project'
        self.artifacts = self.root/'artifacts'
        self.artifacts.mkdir(parents=True)
        self.public = Path(self.temp.name)/'public'
        self.public.mkdir()
        self.git('init', '--quiet')
        self.git('-c','user.name=Test','-c','user.email=synthetic',
                 'commit','--quiet','--allow-empty','-m','synthetic parent')
        self.parent = self.git('rev-parse','HEAD').decode().strip()
        self.stage = self.artifacts/'synthetic-panel'
        self.deadline = (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
        self.helper = Path(self.temp.name)/'standalone-helper.py'
        self.helper.write_bytes(b'synthetic helper source\n')
        self.paths = {n:self.root/'models'/(n+'.npz') for n in ('merged','unweighted','crop')}
        self.opponents = {n+'.py':self.root/'opponents'/(n+'.py') for n in ('round2','market','teacher')}
        for path in (*self.paths.values(), *self.opponents.values()):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('synthetic '+path.name).encode())
        hashes = {name:p.old.sha(path) for name,path in self.paths.items()}
        parent_cfg = dict(engine_commit=e.ENGINE_COMMIT, engine_sha256=e.ENGINE_SHA256,
                          official_loader_sha256='a'*64)
        env = {'interpreter_sha256':'b'*64, 'versions':{'python':'synthetic'}}
        previous = {'environment':env, 'game_limits':{'first':1,'second':2}}
        reference = dict(runtime_files={'old.py':'c'*64}, public_runtime_sha256={'old.py':'d'*64},
                         new_source_sha256={'prev.py':'e'*64}, historical_evidence_sha256={'prev.json':'f'*64})
        freeze = dict(reference, parent_commit=self.parent, supplemental_runtime_sha256={})
        self.inputs = (parent_cfg,self.paths,hashes,self.opponents,previous,env,reference,freeze)
        self.prior = SeedInventory([1000000000,1000000001,1000000002,1000000003], 0, 0)
        self.inventory = SeedInventory(self.prior.private_values(), 2, 20, self.parent)
        self.events = []

        def inventory(stage):
            self.assertFalse(Path(stage).exists())
            self.events.append('inventory')
            return self.inventory

        self.stack = [patch.object(p.old,'ROOT',self.root),patch.object(e,'ROOT',self.root),
            patch.object(p.prior_v2_guard,'require_v2_consumed_chain',return_value=self.prior),
            patch.object(p.old,'PUBLIC',self.public),patch.object(p.old,'trusted_artifacts',return_value=self.artifacts),
            patch.object(p.rescan,'preselection_inventory',side_effect=inventory),
            patch.object(p,'_inputs',side_effect=lambda parent:copy.deepcopy(self.inputs)),
            patch.object(e,'_validate_sources'),
            patch.object(p.secrets,'token_hex',side_effect=['67'*32,'ab89'*16]),
            patch.object(p.secrets,'randbelow',side_effect=self.randbelow)]
        for item in self.stack:
            item.start()
            self.addCleanup(item.stop)
        self.draws = iter([0, 1, p.evaluation.GATES['bootstrap_seed']-1000000000, 10,10,11,12,13])

    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.public),*args], stderr=subprocess.PIPE)

    def randbelow(self,_):
        self.events.append('random')
        self.assertTrue((self.stage/r.MARKER).is_file())
        self.assertTrue((self.stage/r.SNAPSHOT).is_file())
        return next(self.draws)

    def prepare(self):
        return p.prepare(self.stage,self.deadline,self.helper,run_kind=e.FORMAL)

    def review(self,receipt):
        return dict(receipt, reviewer='/root/synthetic_reviewer', review_basis='independent_subagent',
                    passed=True, open_findings=[], checks=['source_identity'])

    def publish(self):
        dest = self.public/'review'/'evidence'/self.stage.name
        dest.mkdir(parents=True)
        for name in (*r.PUBLIC_FILES,r.SAFETY):
            (dest/name).write_bytes((self.stage/name).read_bytes())
        self.git('add','review/evidence/'+self.stage.name)
        self.git('-c','user.name=Test','-c','user.email=synthetic',
                 'commit','--quiet','-m','synthetic registration')
        return self.git('rev-parse','HEAD').decode().strip()

    def check_bound_documents(self, stage, commit):
        tracked=[n for n in p.old.git('ls-files','-z').decode().split('\0') if n]
        files,_ = r._public_evidence(stage,commit,tracked)
        r._verify_documents(stage,self.parent,commit,files)
        return {}

    def test_snapshot_precedes_selection_and_registration_binds_real_git_child(self):
        receipt = self.prepare()
        self.assertEqual(self.events[0],'inventory')
        self.assertEqual(self.events.count('random'),8)
        document = json.loads((self.stage/r.PRIVATE).read_text(encoding='utf-8'))
        self.assertEqual(document['seeds'],[1000000010,1000000011,1000000012,1000000013])
        self.assertEqual(json.loads((self.stage/'registration.json').read_text())['game_limits'],{'first':1,'second':2})
        p.accept_review(self.stage,self.review(receipt),self.helper)
        commit = self.publish()
        with patch.object(e.Context,'ready',autospec=True,
                          side_effect=lambda ctx:self.check_bound_documents(ctx.OUT,ctx.prereg_commit)):
            bound = p.bind_execution(self.stage,commit)
        self.assertEqual(bound['commit'],commit)
        self.assertTrue((self.stage/r.EXECUTION).is_file())
        self.assertFalse((self.stage/'results').exists())
        with self.assertRaises(ValueError): p.bind_execution(self.stage,commit)
        public_text = ''.join((self.stage/name).read_text() for name in (*r.PUBLIC_FILES,r.SAFETY))
        self.assertNotIn(document['salt'],public_text)
        for seed in document['seeds']: self.assertNotIn(str(seed),public_text)

    def test_existing_stage_is_never_adopted_and_no_rng_is_called(self):
        self.stage.mkdir()
        (self.stage/'keep').write_text('unchanged')
        with self.assertRaises(ValueError): self.prepare()
        self.assertEqual((self.stage/'keep').read_text(),'unchanged')
        self.assertNotIn('random',self.events)

    def test_inventory_failure_and_expired_deadline_precede_stage_creation(self):
        with patch.object(p.rescan,'preselection_inventory',side_effect=ValueError('unsafe history')):
            with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.stage.exists())
        with self.assertRaises(ValueError): p.prepare(self.stage,'2000-01-01T00:00:00+00:00',self.helper,run_kind=e.FORMAL)
        self.assertFalse(self.stage.exists())
        self.assertNotIn('random',self.events)

    def test_omitted_failed_v2_role_prevents_selection(self):
        self.inventory = SeedInventory([1000000000,1000000001,1000000002],2,20,self.parent)
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.stage.exists())
        self.assertNotIn('random',self.events)

    def test_review_must_match_identities_and_have_no_open_findings(self):
        receipt=self.prepare()
        review=self.review(receipt)
        for changes in ({'passed':False}, {'review_basis':'webchat'}, {'reviewer':'/root/'},
                        {'open_findings':['unresolved']}, {'registration_sha256':'0'*64}, {'checks':[]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                p.accept_review(self.stage,{**review,**changes},self.helper)
        self.assertFalse((self.stage/'pre-review.json').exists())
        p.accept_review(self.stage,review,self.helper)

    def test_review_cannot_expose_salt_even_as_a_valid_reviewer_identifier(self):
        receipt=self.prepare()
        doc=json.loads((self.stage/r.PRIVATE).read_text(encoding='utf-8'))
        review=self.review(receipt)
        review['reviewer']='/root/'+doc['salt']
        self.assertTrue(e.valid_public_review(review))
        with self.assertRaises(ValueError): p.accept_review(self.stage,review,self.helper)
        self.assertFalse((self.stage/'pre-review.json').exists())
        with self.assertRaises(ValueError):
            p._redacted({'text':doc['salt'].upper()},doc['seeds'],doc['salt'])

    def test_review_rejects_seed_embedded_in_a_longer_numeric_identifier(self):
        receipt=self.prepare()
        doc=json.loads((self.stage/r.PRIVATE).read_text(encoding='utf-8'))
        good=self.review(receipt)
        for reviewer in ('/root/reviewer_9'+str(doc['seeds'][0]),
                         '/root/reviewer_'+str(doc['seeds'][0])+'1'):
            review={**good,'reviewer':reviewer}
            self.assertTrue(e.valid_public_review(review))
            with self.assertRaises(ValueError):
                p.accept_review(self.stage,review,self.helper)
        self.assertFalse((self.stage/'pre-review.json').exists())

    def test_stage_name_containing_selected_seed_stops_before_public_drafts(self):
        self.stage=self.artifacts/'synthetic-panel-910000000101'
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertTrue((self.stage/r.PRIVATE).is_file())
        self.assertFalse((self.stage/'root-decision.json').exists())
        self.assertFalse((self.stage/'pre-review.json').exists())

    def test_safety_redaction_failure_precedes_review_write(self):
        receipt=self.prepare()
        real_redacted=p._redacted
        def reject_safety(value,seeds,salt):
            if isinstance(value,dict) and value.get('schema') == r.SCHEMA and 'evidence_sha256' in value:
                raise ValueError('synthetic safety redaction failure')
            return real_redacted(value,seeds,salt)
        with patch.object(p,'_redacted',side_effect=reject_safety):
            with self.assertRaises(ValueError):
                p.accept_review(self.stage,self.review(receipt),self.helper)
        self.assertFalse((self.stage/'pre-review.json').exists())
        self.assertFalse((self.stage/r.SAFETY).exists())

    def test_review_rejects_paths_and_unapproved_public_check_text(self):
        receipt=self.prepare()
        good=self.review(receipt)
        separator=chr(92)
        drive_path='D:'+separator+'Users'+separator+'example'+separator+'private'
        network_path=separator*2+'example'+separator+'share'+separator+'private'
        posix_path='/'+'home/'+'example/private'
        invalid=(
            {'reviewer':'/root/'+drive_path},
            {'reviewer':'/root/..'+posix_path},
            {'checks':[drive_path]},
            {'checks':[network_path]},
            {'checks':['looked at '+posix_path]},
            {'checks':['source_identity','source_identity']},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                p.accept_review(self.stage,{**good,**changes},self.helper)
        self.assertFalse((self.stage/'pre-review.json').exists())
        p.accept_review(self.stage,good,self.helper)

    def test_tampered_draft_opponent_or_marker_prevents_review(self):
        receipt=self.prepare()
        review=self.review(receipt)
        for path in (self.stage/'registration.json', self.stage/'sources'/'market.py', self.stage/r.MARKER):
            original=path.read_bytes()
            path.write_bytes(original+b' ')
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                p.accept_review(self.stage,review,self.helper)
            path.write_bytes(original)
        self.assertFalse((self.stage/'pre-review.json').exists())
        p.accept_review(self.stage,review,self.helper)

    def test_unexpected_files_prevent_review(self):
        receipt=self.prepare()
        (self.stage/'unexpected').write_text('do not adopt')
        with self.assertRaises(ValueError): p.accept_review(self.stage,self.review(receipt),self.helper)
        self.assertFalse((self.stage/'pre-review.json').exists())

    def test_non_child_commit_and_changed_public_evidence_are_rejected(self):
        receipt=self.prepare()
        p.accept_review(self.stage,self.review(receipt),self.helper)
        with self.assertRaises(ValueError): p.bind_execution(self.stage,self.parent)
        self.assertFalse((self.stage/r.EXECUTION).exists())
        commit=self.publish()
        (self.stage/'pre-review.json').write_text('{}',encoding='utf-8')
        with self.assertRaises(ValueError): p.bind_execution(self.stage,commit)
        self.assertFalse((self.stage/r.EXECUTION).exists())

    def test_failed_post_bind_validation_preserves_receipt_and_refuses_retry(self):
        receipt=self.prepare()
        p.accept_review(self.stage,self.review(receipt),self.helper)
        commit=self.publish()
        with patch.object(e.Context,'ready',side_effect=ValueError('synthetic failed readiness')):
            with self.assertRaises(ValueError): p.bind_execution(self.stage,commit)
        self.assertTrue((self.stage/r.EXECUTION).is_file())
        with self.assertRaises(ValueError): p.bind_execution(self.stage,commit)

    def test_registration_commit_cannot_include_unrelated_changes(self):
        receipt=self.prepare()
        p.accept_review(self.stage,self.review(receipt),self.helper)
        dest=self.public/'review'/'evidence'/self.stage.name
        dest.mkdir(parents=True)
        for name in (*r.PUBLIC_FILES,r.SAFETY):
            (dest/name).write_bytes((self.stage/name).read_bytes())
        (self.public/'unrelated.txt').write_text('other change')
        self.git('add','review/evidence/'+self.stage.name,'unrelated.txt')
        self.git('-c','user.name=Test','-c','user.email=synthetic',
                 'commit','--quiet','-m','synthetic registration')
        commit=self.git('rev-parse','HEAD').decode().strip()
        with self.assertRaises(ValueError): p.bind_execution(self.stage,commit)
        self.assertFalse((self.stage/r.EXECUTION).exists())

    def test_changed_standalone_helper_prevents_review(self):
        receipt = self.prepare()
        self.helper.write_bytes(b'different helper source\n')
        with self.assertRaises(ValueError): p.accept_review(self.stage,self.review(receipt),self.helper)
        self.assertFalse((self.stage/'pre-review.json').exists())


if __name__ == '__main__':
    unittest.main()
