"""Synthetic v2 failure-chain checks; no real seeds, replay or games."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from . import failed_history as guard
from learning.task_anchor_once_v1.storage import owned_writes


def encoded(value):
    return (json.dumps(value,sort_keys=True)+'\n').encode('utf-8')


class FailedHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.artifacts, self.public = root/'artifacts', root/'public'
        self.stage = self.artifacts/guard.STAGE_NAME
        self.stage.mkdir(parents=True)
        self.public.mkdir()
        self.evidence = self.public/guard.EVIDENCE_PREFIX
        self.evidence.mkdir(parents=True)
        self.failure_commit = 'c'*40
        self.head = 'd'*40
        self.clean = True
        self.ancestry = True
        self.document = {'version':1,'seeds':[101,202,303,404],'salt':'ab'*32}
        self.private_raw = encoded(self.document)
        self.cfg = {'seed_count':4, 'seed_document_sha256':guard._digest(self.private_raw),
            'seed_commitment':guard.old.seed_commitment(self.document),
            'seed_commitment_domain':guard.old.SEED_DOMAIN.decode('ascii').rstrip('\x00')}
        self.registration = encoded(self.cfg)
        self.freeze = encoded({'registration_sha256':guard._digest(self.registration)})
        reg_sha, freeze_sha = guard._digest(self.registration), guard._digest(self.freeze)
        game, wrapper = guard.FIRST_GAME, 'e'*64
        self.receipts = {
            'registration':self.registration,
            'freeze':self.freeze,
            'binding':encoded(dict(commit=guard.REGISTRATION_COMMIT,registration_sha256=reg_sha,freeze_sha256=freeze_sha)),
            'campaign_failure':encoded(dict(attempted_games=1,completed_games=0,
                                             current_game_additional_transitions='unknown',retry_allowed=False)),
            'coordinator_failure':encoded(dict(error_kind='CampaignFailure',cleanup_state='confirmed_empty',retry_allowed=False)),
            'launcher_result':encoded(dict(passed=False,exit_code=1,child_reaped=True,job_members_zero=True,
                                            handles_closed=True,cleanup_known=True,scratch_retained=False,retry_allowed=False)),
            'job_supervision':encoded(dict(game=game,error_kind=None,cleanup_state='recorded',retry_allowed=False,
                supervision=dict(passed=False,exit_code=1,launcher_reaped=True,job_members_zero=True,
                                 cleanup_error_kind=None,stop_reason=None))),
            'worker_storage':encoded(dict(passed=False,error_kind='ValueError',
                                           output=dict(used_bytes=0,created_count=0,refused=0))),
            'canonical':encoded(dict(game=game,wrapper_sha256=wrapper,passed=True,launcher_reaped=True,
                                     job_members_zero=True,isolated_cache_prefix_absent=True,retry_allowed=False)),
            'job_start':encoded(dict(game=game,wrapper_sha256=wrapper,retry_allowed=False)),
        }
        self.cert = dict(schema='bounded-panel-failure-v1',registration_commit=guard.REGISTRATION_COMMIT,
            source_commit=guard.SOURCE_COMMIT,status='failed-no-partial-economic-evaluation',
            seed_roles_reserved_no_reuse=4,attempted_games=1,accepted_games=0,retry_allowed=False,
            economic_evaluation_started=False,partial_results_used_for_selection=False,
            local_receipt_sha256={key:guard._digest(raw) for key,raw in self.receipts.items()})
        self.write_fixture()
        self.patches = [patch.object(guard.old,'PUBLIC',self.public),
                        patch.object(guard.old,'trusted_artifacts',return_value=self.artifacts),
                        patch.object(guard,'_git',side_effect=self.git)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def write_fixture(self):
        for key,rel in guard.RECEIPT_PATHS.items():
            path=self.stage/rel
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(self.receipts[key])
        (self.stage/'private-seeds.local.json').write_bytes(self.private_raw)
        (self.evidence/'registration.json').write_bytes(self.registration)
        (self.evidence/'freeze.json').write_bytes(self.freeze)
        (self.evidence/'failure-certificate.json').write_bytes(encoded(self.cert))

    def git(self,*args):
        if args[:3] == ('rev-list','--parents','-n'):
            return (guard.REGISTRATION_COMMIT+' '+guard.SOURCE_COMMIT+'\n').encode()
        if args[:2] == ('merge-base','--is-ancestor'):
            if not self.ancestry: raise ValueError('synthetic ancestry failure')
            return b''
        if args == ('rev-parse','HEAD'):
            return (self.head+'\n').encode()
        if args == ('status','--porcelain'):
            return b'' if self.clean else b'synthetic dirty state'
        if args[0] == 'show':
            revision,path = args[1].split(':',1)
            if revision == guard.REGISTRATION_COMMIT and path == guard.EVIDENCE_PREFIX+'registration.json':
                return self.registration
            if revision == guard.REGISTRATION_COMMIT and path == guard.EVIDENCE_PREFIX+'freeze.json':
                return self.freeze
            if revision == self.failure_commit and path == guard.EVIDENCE_PREFIX+'failure-certificate.json':
                return encoded(self.cert)
        raise ValueError('synthetic unrecognized Git request')

    def load(self):
        return guard.load_failed_v2(self.failure_commit,self.head)

    def assert_rejected(self):
        with self.assertRaisesRegex(ValueError,'^Prior v2 reserved-seed evidence is invalid or unavailable$'):
            self.load()

    def test_bound_roles_are_redacted_and_reject_collision(self):
        inventory = self.load()
        self.assertEqual(inventory.count,4)
        self.assertEqual(repr(inventory),'SeedInventory(count=4)')
        self.assertTrue(all(inventory.contains(seed) for seed in self.document['seeds']))
        with self.assertRaises(ValueError): inventory.require_unused([101,500])
        inventory.require_unused([501,502])

    def test_unpinned_or_non_ancestor_failure_never_passes(self):
        for failure,head in ((None,self.head),('c'*39,self.head),(self.failure_commit,None)):
            with self.subTest(failure=failure,head=head), self.assertRaises(ValueError):
                guard.load_failed_v2(failure,head)
        self.ancestry=False
        self.assert_rejected()
        self.ancestry=True
        self.clean=False
        self.assert_rejected()

    def test_changed_private_document_or_local_receipt_is_rejected(self):
        for path in (self.stage/'private-seeds.local.json',self.stage/guard.RECEIPT_PATHS['worker_storage'],
                     self.stage/guard.RECEIPT_PATHS['registration']):
            original=path.read_bytes()
            path.write_bytes(original+b' ')
            with self.subTest(name=path.name): self.assert_rejected()
            path.write_bytes(original)

    def test_public_certificate_or_registration_byte_change_is_rejected(self):
        for name in ('registration.json','freeze.json','failure-certificate.json'):
            path=self.evidence/name
            original=path.read_bytes()
            path.write_bytes(original+b' ')
            with self.subTest(name=name): self.assert_rejected()
            path.write_bytes(original)

    def test_certificate_rejects_wrong_roles_status_or_boolean_numeric_values(self):
        for change in ({'seed_roles_reserved_no_reuse':3},{'accepted_games':1},{'attempted_games':True},
                       {'retry_allowed':True},{'economic_evaluation_started':0},
                       {'registration_commit':'f'*40},{'status':'success'}):
            original=copy.deepcopy(self.cert)
            self.cert.update(change)
            (self.evidence/'failure-certificate.json').write_bytes(encoded(self.cert))
            with self.subTest(change=change): self.assert_rejected()
            self.cert=original
            (self.evidence/'failure-certificate.json').write_bytes(encoded(self.cert))

    def test_valid_hashes_do_not_override_failed_cleanup_or_missing_commitment(self):
        for key,updates in (('launcher_result',{'cleanup_known':False}),
                            ('campaign_failure',{'current_game_additional_transitions':0}),
                            ('worker_storage',{'error_kind':'OtherError'})):
            original=self.receipts[key]
            changed=json.loads(original)
            changed.update(updates)
            self.receipts[key]=encoded(changed)
            self.cert['local_receipt_sha256'][key]=guard._digest(self.receipts[key])
            self.write_fixture()
            with self.subTest(key=key): self.assert_rejected()
            self.receipts[key]=original
            self.cert['local_receipt_sha256'][key]=guard._digest(original)
            self.write_fixture()

    def test_stale_seed_commitment_cannot_be_hidden_by_new_document_hash(self):
        self.document['salt']='cd'*32
        self.private_raw=encoded(self.document)
        self.cfg['seed_document_sha256']=guard._digest(self.private_raw)
        self.registration=encoded(self.cfg)
        self.freeze=encoded({'registration_sha256':guard._digest(self.registration)})
        self.receipts['registration']=self.registration
        self.receipts['freeze']=self.freeze
        self.receipts['binding']=encoded(dict(commit=guard.REGISTRATION_COMMIT,
            registration_sha256=guard._digest(self.registration),freeze_sha256=guard._digest(self.freeze)))
        for key in ('registration','freeze','binding'):
            self.cert['local_receipt_sha256'][key]=guard._digest(self.receipts[key])
        self.write_fixture()
        self.assert_rejected()


@unittest.skipUnless((guard.old.PUBLIC/'.git').exists() and shutil.which('git'),
                     'The local public Git repository is unavailable')
class BudgetCompatibilityTests(unittest.TestCase):
    def test_real_git_identity_check_works_inside_existing_output_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            with owned_writes(Path(temp),0):
                self.assertEqual(guard._git('rev-parse','--is-inside-work-tree').strip(),b'true')
                with self.assertRaisesRegex(ValueError,'^Prior v2 Git identity unavailable$'):
                    guard._git('rev-parse','--verify','synthetic-absent-revision')


if __name__ == '__main__':
    unittest.main()
