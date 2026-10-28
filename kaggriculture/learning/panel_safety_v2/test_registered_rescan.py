import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import registered_rescan as r


def encoded(value):
    return (json.dumps(value, sort_keys=True) + '\n').encode()


class RegisteredRescanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root/'artifacts'
        self.public = self.root/'public'
        self.artifacts.mkdir()
        self.public.mkdir()
        self.stage = self.artifacts/'fresh-stage'
        self.stage.mkdir()
        self.parent = 'a'*40
        self.commit = 'b'*40
        self.seeds = [1234567001,1234567002,1234567003,1234567004]
        self.document = {'version':1,'seeds':self.seeds,'salt':'cd'*32}
        self.snapshot = {'excluded_uint32':[101,202,303]}
        self.marker = {'schema':r.SCHEMA,'stage':self.stage.name,'parent_commit':self.parent,
                       'created_exclusively':True,'nonce':'ab'*32}
        self.prefix = 'review/evidence/'+self.stage.name+'/'
        self.tracked = []
        self.parent_paths = []
        self.git_head = self.commit
        self.git_status = b''
        self.blobs = {}
        self._registration()
        self.patches = [patch.object(r.old,'trusted_artifacts',return_value=self.artifacts),
                        patch.object(r.old,'PUBLIC',self.public), patch.object(r.old,'git',side_effect=self.git),
                        patch.object(r.history,'load_consumed_prior',return_value=r.history.SeedInventory([901,902,903,904],0,0))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else encoded(value))

    def _registration(self, cfg_override=None, decision_override=None):
        self.write(self.stage/r.PRIVATE,self.document)
        self.write(self.stage/r.SNAPSHOT,self.snapshot)
        self.write(self.stage/r.MARKER,self.marker)
        seed_sha = r._sha((self.stage/r.PRIVATE).read_bytes())
        snapshot_sha = r._sha((self.stage/r.SNAPSHOT).read_bytes())
        commitment = r.old.seed_commitment(self.document)
        self.cfg = dict(baseline_head=self.parent, seed_count=4, seed_document_sha256=seed_sha,
            seed_commitment=commitment, seed_commitment_domain=r.old.SEED_DOMAIN.decode().rstrip('\x00'),
            seed_inventory_sha256=snapshot_sha, opponents=['round2','market','teacher'],
            policies=['merged','unweighted','teacher_task'], seats=[0,1],
            merged_checkpoint_sha256='1'*64, unweighted_checkpoint_sha256='2'*64,
            source_sha256={name+'.py':r._sha((name+' source').encode()) for name in ('round2','market','teacher')})
        self.cfg.update(cfg_override or {})
        decision = dict(previous_commit=self.parent,status='approved_for_implementation_not_execution',
                        seed_count=4,seed_document_sha256=seed_sha,seed_commitment=commitment)
        decision.update(decision_override or {})
        files = {'root-decision.json':encoded(decision),'registration.json':encoded(self.cfg)}
        freeze = dict(parent_commit=self.parent,root_decision_sha256=r._sha(files['root-decision.json']),
                      registration_sha256=r._sha(files['registration.json']))
        files['freeze.json'] = encoded(freeze)
        review = dict(baseline_commit=self.parent,passed=True,root_decision_sha256=freeze['root_decision_sha256'],
            registration_sha256=freeze['registration_sha256'],freeze_sha256=r._sha(files['freeze.json']),
            seed_document_sha256=seed_sha,seed_commitment=commitment,seed_inventory_sha256=snapshot_sha)
        files['pre-review.json'] = encoded(review)
        self.write(self.stage/r.EXECUTION,dict(commit=self.commit,
            registration_sha256=freeze['registration_sha256'],freeze_sha256=r._sha(files['freeze.json'])))
        safety = dict(schema=r.SCHEMA,stage=self.stage.name,parent_commit=self.parent,
            marker_sha256=r._sha((self.stage/r.MARKER).read_bytes()),
            evidence_sha256={name:r._sha(raw) for name,raw in files.items()})
        files[r.SAFETY] = encoded(safety)
        self.blobs.clear()
        self.tracked = []
        for name,raw in files.items():
            self.write(self.stage/name,raw)
            self.write(self.public/self.prefix/name,raw)
            self.tracked.append(self.prefix+name)
            self.blobs[self.commit+':'+self.prefix+name] = raw
        for name in ('round2','market','teacher'):
            self.write(self.stage/'sources'/(name+'.py'),(name+' source').encode())

    def git(self,*args):
        if args[:2] == ('rev-parse','HEAD'): return self.git_head.encode()
        if args[:2] == ('status','--porcelain'): return self.git_status
        if args[0] == 'rev-list': return (self.commit+' '+self.parent).encode()
        if args[0] == 'merge-base': return b''
        if args[0] == 'ls-tree': return ('\0'.join(self.parent_paths)+'\0').encode()
        if args[0] == 'ls-files': return ('\0'.join(self.tracked)+'\0').encode()
        if args[0] == 'show': return self.blobs[args[1]]
        raise AssertionError('Unexpected synthetic Git call')

    def check(self):
        return r.rescan_registered(self.stage,self.commit)

    def job(self):
        plan=r.plans(self.seeds)[0]
        bot=b'synthetic bot'; self.write(self.stage/'agents'/(plan['game']+'.py'),bot)
        item=dict(plan,bot_sha256=r._sha(bot),rival_sha256=self.cfg['source_sha256'][plan['opponent']+'.py'],
                  task_checkpoint_sha256=self.cfg[plan['policy']+'_checkpoint_sha256'])
        path=self.stage/'job-inputs'/(plan['game']+'.json'); self.write(path,item)
        return path,item

    def test_registered_stage_and_valid_job_do_not_self_collide(self):
        self.job()
        inventory=self.check()
        self.assertTrue(inventory.contains(901))
        self.assertTrue(inventory.contains(101))
        self.assertNotIn(str(self.seeds[0]),repr(inventory))
        self.assertNotIn(self.document['salt'],repr(inventory))

    def test_added_historical_seed_is_rejected(self):
        path='review/evidence/unrelated/registration.json'
        self.write(self.public/path,{'reserved_seeds':[self.seeds[0]]})
        self.tracked.append(path)
        with self.assertRaises(ValueError) as raised: self.check()
        self.assertNotIn(str(self.seeds[0]),str(raised.exception))

    def test_historical_float_alias_index_and_decimal_string_collisions(self):
        path=self.artifacts/'prior'/'registration.json'
        for value in ({'test_seed':float(self.seeds[0])}, {'test_seeds':{'S01':self.seeds[0]}},
                      {'test_seeds':{'first_game':self.seeds[0]}},
                      {'test_seeds':{'0':self.seeds[0]}}, {'test_seed':str(self.seeds[0])},
                      {'test_seed':str(self.seeds[0])+'.0'}, {'seed_map':{'first_game':self.seeds[0]}},
                      {'test_seed':str(self.seeds[0]).translate(str.maketrans('0123456789','０１２３４５６７８９'))},
                      {'seed_map':{'round2':[self.seeds[0]]}}, {'test_seeds':{'market':[self.seeds[0]]}},
                      {'testSeeds':{'firstGame':self.seeds[0]}},
                      {'test_seeds':{'S01':{'value':self.seeds[0]}}},
                      {'test_seeds':{str(self.seeds[0]):{'score':999}}}):
            with self.subTest(kind=next(iter(value))):
                self.write(path,value)
                with self.assertRaises(ValueError) as raised: self.check()
                self.assertNotIn(str(self.seeds[0]),str(raised.exception))

    def test_count_hash_boolean_and_unrelated_metadata_are_not_seeds(self):
        path=self.artifacts/'prior'/'registration.json'
        self.write(path,{'seed_count':self.seeds[0],'num_seeds':self.seeds[1],
            'seed_sha256':str(self.seeds[2]),'score':self.seeds[3],
            'seed_hash_algorithm':str(self.seeds[0]),'seed_score':self.seeds[1],
            'test_seeds':{'S01':True},'seed_selection':{'count':self.seeds[0],
            'estimated_score':self.seeds[1]}})
        inventory=self.check()
        self.assertFalse(any(inventory.contains(value) for value in self.seeds))

    def test_noncolliding_mapped_values_are_added_to_inventory(self):
        path=self.artifacts/'prior'/'registration.json'
        self.write(path,{'test_seeds':{'S01':1111.0,'2':2222,'first_game':'3333'}})
        inventory=self.check()
        self.assertTrue(all(inventory.contains(value) for value in (2,1111,2222,3333)))

    def test_nonfinite_or_fractional_number_in_seed_slot_fails_closed(self):
        path=self.artifacts/'prior'/'registration.json'
        for raw in (b'{"test_seed":1e100000}', b'{"test_seeds":{"S01":1.5}}'):
            with self.subTest(raw=raw):
                self.write(path,raw)
                with self.assertRaises(ValueError): self.check()
        self.write(path,b'{"score":1e100000}')
        self.check()

    def test_oversized_numeric_seed_text_fails_closed(self):
        self.write(self.artifacts/'prior'/'registration.json', {'test_seed':'0'*129+str(self.seeds[0])})
        with self.assertRaises(ValueError): self.check()

    def test_private_whitelisted_history_in_both_roots_is_scanned(self):
        for root in (self.artifacts/'prior'/'private',self.public/'private'):
            for name in ('registration.json','seed-inventory.local.json'):
                with self.subTest(name=name,location=root.name):
                    path=root/name
                    self.write(path,{'test_seed':self.seeds[0]})
                    with self.assertRaises(ValueError): self.check()
                    path.unlink()

    def test_private_job_inputs_are_included_in_both_roots(self):
        for root in (self.artifacts/'prior'/'private', self.public/'private'):
            with self.subTest(location=root.name):
                path = root/'job-inputs'/'game.json'
                self.write(path, {'seed':self.seeds[0]})
                with self.assertRaises(ValueError): self.check()
                path.unlink()

    def test_ignored_own_public_metadata_is_rejected(self):
        self.write(self.public/self.prefix/'private-seeds.local.json',{'test_seed':self.seeds[0]})
        with self.assertRaises(ValueError): self.check()

    def test_own_public_extra_directory_or_non_json_file_is_rejected(self):
        extra = self.public/self.prefix/'results'/'registration.json'
        self.write(extra, {'test_seed':self.seeds[0]})
        with self.assertRaises(ValueError): self.check()
        extra.unlink()
        extra.parent.rmdir()
        self.write(self.public/self.prefix/'extra.txt', b'unknown')
        with self.assertRaises(ValueError): self.check()

    def test_real_git_ignored_public_seed_metadata_is_scanned(self):
        def git(*args):
            return subprocess.run(['git','-C',str(self.public),*args],capture_output=True,check=True)
        git('init','--quiet')
        self.write(self.public/'.gitignore',b'private-seeds.local.json\n')
        git('add','.gitignore','review/evidence/fresh-stage')
        git('-c','user.name=Test','-c','user.email='+'test'+'@'+'example.invalid',
            'commit','--quiet','-m','fixture')
        path='review/evidence/prior/private-seeds.local.json'
        self.write(self.public/path,{'test_seed':self.seeds[0]})
        git('check-ignore','-q','--',path)
        self.assertEqual(git('ls-files','--',path).stdout,b'')
        self.assertEqual(git('status','--porcelain','--untracked-files=all').stdout,b'')
        with self.assertRaises(ValueError): self.check()

    def test_preselection_snapshot_keeps_expanded_history_after_sources_deleted(self):
        def git(*args):
            return subprocess.run(['git','-C',str(self.public),*args],capture_output=True,check=True)
        git('init','--quiet')
        self.write(self.public/'.gitignore',b'private/\nprivate-seeds.local.json\n')
        git('add','.gitignore','review/evidence/fresh-stage')
        git('-c','user.name=Test','-c','user.email='+'test'+'@'+'example.invalid',
            'commit','--quiet','-m','fixture')
        values=(2234567001,2234567002,2234567003,2234567004)
        files=(self.artifacts/'prior'/'registration.json',self.public/'private'/'registration.json',
               self.artifacts/'prior'/'private'/'seed-inventory.local.json',
               self.public/'review'/'evidence'/'prior'/'private-seeds.local.json')
        self.write(files[0],{'test_seeds':{'S01':float(values[0])}})
        self.write(files[1],{'test_seeds':{'first_game':values[1]}})
        self.write(files[2],{'test_seeds':{'0':values[2]}})
        self.write(files[3],{'test_seed':values[3]})
        relative='review/evidence/prior/private-seeds.local.json'
        git('check-ignore','-q','--',relative)
        self.assertEqual(git('ls-files','--',relative).stdout,b'')
        self.assertEqual(git('status','--porcelain','--untracked-files=all').stdout,b'')
        snapshot=r.preselection_inventory(self.artifacts/'next-stage')
        for path in files: path.unlink()
        for value in values:
            with self.assertRaises(ValueError): snapshot.require_unused([value])
        self.assertNotIn(str(values[0]),repr(snapshot))

    def test_preselection_detects_changes_between_expanded_scans(self):
        real=r._expanded_history
        calls=[]
        def scan(*args):
            result=real(*args); calls.append(1)
            if len(calls)==1:
                self.write(self.artifacts/'other-stage'/'registration.json',{'test_seed':1234567001})
            return result
        with patch.object(r,'_expanded_history',side_effect=scan):
            with self.assertRaises(ValueError): r.preselection_inventory(self.artifacts/'next-stage')

    def test_preselection_detects_parent_change_after_scan_starts(self):
        real=r._expanded_history
        calls=[]
        def scan(*args):
            result=real(*args); calls.append(1)
            if len(calls)==1: self.git_head='c'*40
            return result
        with patch.object(r,'_expanded_history',side_effect=scan):
            with self.assertRaises(ValueError): r.preselection_inventory(self.artifacts/'next-stage')

    def test_frozen_history_and_consumed_four_are_never_dropped(self):
        self.snapshot={'excluded_uint32':[self.seeds[0]]}; self._registration()
        with self.assertRaises(ValueError): self.check()
        self.snapshot={'excluded_uint32':[101]}; self._registration()
        with patch.object(r.history,'load_consumed_prior',return_value=r.history.SeedInventory([self.seeds[3],901,902,903],0,0)):
            with self.assertRaises(ValueError): self.check()

    def test_owner_and_public_metadata_must_be_exact(self):
        self.marker['created_exclusively']=False; self.write(self.stage/r.MARKER,self.marker)
        with self.assertRaises(ValueError): self.check()
        self.marker['created_exclusively']=True; self._registration()
        self.write(self.stage/'unexpected-seeds.json',{'seeds':[7]})
        with self.assertRaises(ValueError): self.check()
        (self.stage/'unexpected-seeds.json').unlink()
        self.tracked.append(self.prefix+'other.json')
        with self.assertRaises(ValueError): self.check()

    def test_job_inputs_must_match_plan_and_source(self):
        path,item=self.job()
        item['seed']=self.seeds[1]; self.write(path,item)
        with self.assertRaises(ValueError) as raised: self.check()
        self.assertNotIn(str(self.seeds[1]),str(raised.exception))
        path,item=self.job(); item['extra_seed']=999; self.write(path,item)
        with self.assertRaises(ValueError): self.check()
        path,item=self.job(); self.write(self.stage/'agents'/(item['game']+'.py'),b'wrong')
        with self.assertRaises(ValueError): self.check()

    def test_job_boolean_seat_or_float_seed_are_rejected(self):
        path,item=self.job(); item['seed']=float(item['seed']); self.write(path,item)
        with self.assertRaises(ValueError): self.check()
        path,item=self.job(); item['side']=False; self.write(path,item)
        with self.assertRaises(ValueError): self.check()

    def test_registered_boolean_seats_and_float_seed_count_are_rejected(self):
        self._registration(cfg_override={'seats':[False,True]})
        with self.assertRaises(ValueError): self.check()
        self._registration(decision_override={'seed_count':4.0})
        with self.assertRaises(ValueError): self.check()
        self._registration(decision_override={'status':'incorrect'})
        with self.assertRaises(ValueError): self.check()

    def test_git_and_commit_evidence_fail_closed(self):
        self.git_head='c'*40
        with self.assertRaises(ValueError): self.check()
        self.git_head=self.commit; self.git_status=b' M file'
        with self.assertRaises(ValueError): self.check()
        self.git_status=b''; self.parent_paths=[self.prefix+'registration.json']
        with self.assertRaises(ValueError): self.check()
        self.parent_paths=[self.prefix.rstrip('/').upper()]
        with self.assertRaises(ValueError): self.check()
        self.parent_paths=[]; self.write(self.stage/'registration.json',b'{}')
        with self.assertRaises(ValueError): self.check()

    def test_private_commitment_and_snapshot_tampering_fail(self):
        self.write(self.stage/r.PRIVATE,{'version':1,'seeds':self.seeds,'salt':'de'*32})
        with self.assertRaises(ValueError): self.check()
        self._registration(); self.write(self.stage/r.SNAPSHOT,{'excluded_uint32':[9]})
        with self.assertRaises(ValueError): self.check()

    def test_private_document_change_during_history_scan_is_detected(self):
        real = r.history._scan_metadata
        calls=[]
        def scan(*args):
            result=real(*args); calls.append(1)
            if len(calls)==1:
                self.write(self.stage/r.PRIVATE,{'version':1,'seeds':self.seeds,'salt':'de'*32})
            return result
        with patch.object(r.history,'_scan_metadata',side_effect=scan):
            with self.assertRaises(ValueError): self.check()

    def test_new_history_during_scan_does_not_disappear(self):
        real = r.history._scan_metadata
        calls=[]
        def scan(*args):
            result=real(*args); calls.append(1)
            if len(calls)==1:
                self.write(self.artifacts/'other-stage'/'registration.json',{'reserved_seeds':[self.seeds[0]]})
            return result
        with patch.object(r.history,'_scan_metadata',side_effect=scan):
            with self.assertRaises(ValueError): self.check()

    def test_malformed_own_metadata_and_duplicate_json_fail(self):
        path,item=self.job(); path.write_bytes(b'{"seed":1,"seed":2}')
        with self.assertRaises(ValueError): self.check()
        path.unlink(); self.write(self.stage/'nested'/'seed-roles.local.json',{'reserved':[7]})
        with self.assertRaises(ValueError): self.check()
        (self.stage/'nested'/'seed-roles.local.json').unlink()
        self.write(self.stage/'unknown.json',{'reserved':[7]})
        with self.assertRaises(ValueError): self.check()

    def test_results_and_replays_are_not_read(self):
        for name in ('games','traces','results','job-results'):
            self.write(self.stage/name/'registration.json',b'not json')
        self.write(self.stage/'game-ledger.json',b'not json')
        self.check()

    @unittest.skipUnless(os.name == 'nt','Windows junction check')
    def test_stage_junction_is_rejected_and_target_preserved(self):
        target=self.root/'outside'; target.mkdir(); sentinel=target/'sentinel'; sentinel.write_text('keep')
        link=self.artifacts/'linked-stage'
        done=subprocess.run(['cmd','/d','/c','mklink','/J',str(link),str(target)],capture_output=True)
        if done.returncode: self.skipTest('Windows junction creation unavailable')
        try:
            with self.assertRaises(ValueError): r.rescan_registered(link,self.commit)
        finally:
            if link.is_junction(): link.rmdir()
        self.assertEqual(sentinel.read_text(),'keep')


if __name__ == '__main__': unittest.main()
