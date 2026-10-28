import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import seed_inventory as s


def encoded(value):
    return json.dumps(value, sort_keys=True).encode('utf-8')


def evidence(*, document=None, inventory=None, commitment=None):
    document = document or dict(version=1, seeds=[101, 202, 303, 404], salt='ab' * 32)
    inventory = inventory or {'excluded_uint32': [7, 8]}
    document_bytes, inventory_bytes = encoded(document), encoded(inventory)
    cfg = dict(seed_count=4, seed_document_sha256=s._sha(document_bytes),
        seed_inventory_sha256=s._sha(inventory_bytes),
        seed_commitment=commitment if commitment is not None else s.old.seed_commitment(document),
        seed_commitment_domain=s.old.SEED_DOMAIN.decode('ascii').rstrip('\x00'))
    registration = encoded(cfg)
    freeze = encoded({'registration_sha256': s._sha(registration)})
    execution = encoded(dict(commit=s.PRIOR_COMMIT, registration_sha256=s._sha(registration),
                             freeze_sha256=s._sha(freeze)))
    failure = encoded(dict(stop_reason='available_memory_preflight', retry_allowed=False,
        attempted_games=26, completed_games=26, uncertain_current_game=None))
    cert = encoded(dict(status='incomplete_preflight_stop', execution_commit=s.PRIOR_COMMIT,
        registration_sha256=s._sha(registration), freeze_sha256=s._sha(freeze),
        local_failure_sha256=s._sha(failure), stop_reason='available_memory_preflight',
        all_four_seed_roles_consumed=True, retry_allowed=False, planned_games=72,
        attempted_games=26, accepted_games=26, unstarted_games=46,
        next_game_input_present=False, economic_evaluation_started=False))
    return dict(private_document=document_bytes, prior_inventory=inventory_bytes,
        local_registration=registration, local_freeze=freeze, local_failure=failure,
        local_execution=execution, committed_registration=registration,
        committed_freeze=freeze, committed_certificate=cert)


class ConsumedEvidenceTests(unittest.TestCase):
    def test_all_four_prior_roles_survive_and_repr_is_redacted(self):
        consumed = s._verify_consumed(**evidence())
        self.assertEqual(consumed, {101, 202, 303, 404})
        registry = s.SeedInventory(consumed, 1, 100)
        self.assertEqual(registry.count, 4)
        self.assertTrue(registry.contains(404))
        self.assertNotIn('404', repr(registry))
        registry.require_unused([500, 600])
        for rejected in ([404], [500, 500], [True], [-1], []):
            with self.assertRaises(ValueError):
                registry.require_unused(rejected)

    def test_missing_changed_and_duplicate_evidence_fails_closed(self):
        original = evidence()
        for name in ('private_document', 'prior_inventory', 'local_registration', 'local_freeze',
                     'local_failure', 'local_execution', 'committed_registration',
                     'committed_freeze', 'committed_certificate'):
            for replacement in (None, b'{'):
                changed = {**original, name: replacement}
                with self.subTest(field=name, replacement_type=type(replacement).__name__):
                    with self.assertRaises(ValueError):
                        s._verify_consumed(**changed)
        for name in ('private_document', 'prior_inventory', 'local_registration', 'local_freeze',
                     'local_failure', 'committed_registration', 'committed_freeze'):
            with self.subTest(field=name), self.assertRaises(ValueError):
                s._verify_consumed(**{**original, name:original[name] + b' '})
        with self.assertRaises(ValueError):
            s._json(b'{"seed_count":4,"seed_count":4}')

    def test_seed_document_and_inventory_identity_and_commitment(self):
        original = evidence()
        changed_doc = encoded({'version': 1, 'seeds': [101, 202, 303, 404], 'salt': 'cd' * 32})
        with self.assertRaises(ValueError):
            s._verify_consumed(**{**original, 'private_document': changed_doc})
        with self.assertRaises(ValueError):
            s._verify_consumed(**evidence(commitment='0' * 64))
        with self.assertRaises(ValueError):
            s._verify_consumed(**evidence(inventory={'excluded_uint32': [101]}))

    def test_failure_certificate_must_prove_the_entire_batch_consumed(self):
        original = evidence()
        for key, value in (('all_four_seed_roles_consumed', False), ('execution_commit', 'wrong'),
                           ('retry_allowed', True), ('next_game_input_present', True),
                           ('economic_evaluation_started', True), ('accepted_games', 25)):
            cert = json.loads(original['committed_certificate'])
            cert[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                s._verify_consumed(**{**original, 'committed_certificate': encoded(cert)})

    def test_loader_uses_pinned_ancestors_and_committed_blobs(self):
        fixture = evidence()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/s.PRIOR_STAGE
            out.mkdir()
            names = {'private-seeds.local.json':'private_document', 'seed-inventory.local.json':'prior_inventory',
                'registration.json':'local_registration', 'freeze.json':'local_freeze',
                'games-failure.local.json':'local_failure', 'pre-execution-commit.json':'local_execution'}
            for name, key in names.items():
                (out/name).write_bytes(fixture[key])
            blobs = {s.PRIOR_COMMIT + ':' + s.EVIDENCE + 'registration.json': fixture['committed_registration'],
                     s.PRIOR_COMMIT + ':' + s.EVIDENCE + 'freeze.json': fixture['committed_freeze'],
                     s.FAILURE_COMMIT + ':' + s.EVIDENCE + 'failure-certificate.json': fixture['committed_certificate']}
            calls = []
            def git(*args):
                calls.append(args)
                if args[0] == 'merge-base':
                    return b''
                return blobs[args[1]]
            with patch.object(s.old, 'OUT', out), patch.object(s.old, 'trusted_out', return_value=out), \
                 patch.object(s.old, 'git', side_effect=git):
                inventory=s.load_consumed_prior()
                self.assertEqual(inventory.count,4)
                self.assertTrue(all(inventory.contains(value) for value in (101,202,303,404)))
                self.assertNotIn('404',repr(inventory))
                self.assertIn(('merge-base', '--is-ancestor', s.PRIOR_COMMIT, s.FAILURE_COMMIT), calls)
                (out/'private-seeds.local.json').unlink()
                with self.assertRaises(ValueError):
                    s.load_consumed_prior()


class HistoricalScanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts, self.public = self.root/'artifacts', self.root/'public'
        self.artifacts.mkdir()
        self.public.mkdir()
        self.new = self.artifacts/'future-stage'
        self.new.mkdir()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded(value))

    def test_new_stage_skipped_old_included_without_reading_results(self):
        old = self.artifacts/s.PRIOR_STAGE
        self.write(old/'private-seeds.local.json', {'version':1,'seeds':[101,202,303,404],'salt':'ab'*32})
        self.write(old/'job-inputs'/'game.json', {'seed':101,'score':8080})
        self.write(old/'registration.json', {'test_seed':505,'seed_count':17})
        self.write(self.new/'private-seeds.local.json', {'seeds':[999]})
        self.write(self.new/'job-inputs'/'game.json', {'seed':998})
        for folder in ('games','traces','job-results','job-acceptance','results'):
            path=old/folder/'registration.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('not json: must never be read',encoding='utf-8')
        prior_public = 'review/evidence/prior/registration.json'
        own_public = 'review/evidence/future-stage/registration.json'
        self.write(self.public/prior_public, {'reserved_seeds':[606]})
        self.write(self.public/own_public, {'test_seed':997})
        non_json='review/evidence/prior/job-inputs/not-metadata.txt'
        (self.public/non_json).parent.mkdir(parents=True)
        (self.public/non_json).write_text('not json: must never be read',encoding='utf-8')
        values, count, _ = s._scan_metadata(self.artifacts,self.public,self.new,[prior_public,non_json])
        self.assertTrue({101,202,303,404,505,606}.issubset(values))
        self.assertFalse({17,997,998,999,8080} & values)
        self.assertEqual(count,4)

    def test_old_stage_new_stage_boundaries_and_bad_tracked_paths(self):
        for path in (self.artifacts/s.PRIOR_STAGE, self.artifacts/s.PRIOR_STAGE.upper(), self.artifacts, self.root/'outside',
                     self.artifacts/'x'/'..'/'future-stage'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                s._scan_metadata(self.artifacts,self.public,path,[])
        for name in ('../registration.json', '/registration.json', 'x/../registration.json',
                     'C' + ':/registration.json', 'x\\registration.json', 'x/./registration.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                s._scan_metadata(self.artifacts,self.public,self.new,[name])

    def test_reused_public_stage_name_is_rejected_case_insensitively(self):
        own_public='review/evidence/FUTURE-STAGE/registration.json'
        self.write(self.public/own_public, {'test_seed':997})
        with self.assertRaises(ValueError):
            s._scan_metadata(self.artifacts,self.public,self.new,[own_public])

    def test_build_requires_unoccupied_stage_and_clean_public_repository(self):
        future = self.artifacts/'fresh-stage'
        prior = s.SeedInventory([101,202,303,404], 0, 0)
        self.write(self.artifacts/s.PRIOR_STAGE/'private-seeds.local.json', {'seeds':[101,202,303,404]})
        def git(*args):
            if args[0] == 'status': return b''
            if args[0] == 'ls-files': return b''
            if args[0] == 'rev-parse': return b'a'*40
            if args[0] == 'merge-base': return b''
            raise AssertionError('Unexpected Git request')
        with patch.object(s, 'load_consumed_prior', return_value=prior), \
             patch.object(s.old, 'trusted_artifacts', return_value=self.artifacts), \
             patch.object(s.old, 'PUBLIC', self.public), patch.object(s.old, 'git', side_effect=git):
            inventory = s.build_inventory(future)
            self.assertTrue(all(inventory.contains(value) for value in (101,202,303,404)))
            self.assertEqual(inventory.parent_commit, 'a'*40)
            future.mkdir()
            with self.assertRaises(ValueError): s.build_inventory(future)
            future.rmdir()
            (self.public/'review'/'evidence'/'fresh-stage').mkdir(parents=True)
            with self.assertRaises(ValueError): s.build_inventory(future)
        with patch.object(s, 'load_consumed_prior', return_value=prior), \
             patch.object(s.old, 'trusted_artifacts', return_value=self.artifacts), \
             patch.object(s.old, 'PUBLIC', self.public), patch.object(s.old, 'git', return_value=b' M file'):
            with self.assertRaises(ValueError): s.build_inventory(future)

    def test_build_refuses_repository_head_change_during_scan(self):
        future = self.artifacts/'fresh-stage'
        prior = s.SeedInventory([101,202,303,404], 0, 0)
        heads = iter((b'a'*40, b'b'*40))
        def git(*args):
            if args[0] == 'status': return b''
            if args[0] == 'ls-files': return b''
            if args[0] == 'rev-parse': return next(heads)
            if args[0] == 'merge-base': return b''
            raise AssertionError('Unexpected Git request')
        with patch.object(s, 'load_consumed_prior', return_value=prior), \
             patch.object(s.old, 'trusted_artifacts', return_value=self.artifacts), \
             patch.object(s.old, 'PUBLIC', self.public), patch.object(s.old, 'git', side_effect=git):
            with self.assertRaises(ValueError): s.build_inventory(future)

    def test_build_requires_consumed_certificate_in_parent_history(self):
        future = self.artifacts/'fresh-stage'
        prior = s.SeedInventory([101,202,303,404], 0, 0)
        def git(*args):
            if args[0] == 'status': return b''
            if args[0] == 'rev-parse': return b'c'*40
            if args[0] == 'merge-base': raise subprocess.CalledProcessError(1, 'git')
            raise AssertionError('Unexpected Git request')
        with patch.object(s, 'load_consumed_prior', return_value=prior), \
             patch.object(s.old, 'trusted_artifacts', return_value=self.artifacts), \
             patch.object(s.old, 'PUBLIC', self.public), patch.object(s.old, 'git', side_effect=git):
            with self.assertRaises(ValueError): s.build_inventory(future)

    @unittest.skipUnless(os.name == 'nt', 'Windows junction check')
    def test_build_refuses_broken_public_junction_created_during_scan(self):
        future = self.artifacts/'fresh-stage'
        prior = s.SeedInventory([101,202,303,404], 0, 0)
        junction = self.public/'review'/'evidence'/'fresh-stage'
        junction.parent.mkdir(parents=True)
        def git(*args):
            if args[0] == 'rev-parse': return b'a'*40
            return b''
        def scan(*_):
            result = subprocess.run(['cmd','/d','/c','mklink','/J',str(junction),str(self.root/'missing-target')],
                                    capture_output=True,check=False)
            if result.returncode:
                self.skipTest('Windows junction creation is unavailable')
            return set(),0,0
        try:
            with patch.object(s, 'load_consumed_prior', return_value=prior), \
                 patch.object(s.old, 'trusted_artifacts', return_value=self.artifacts), \
                 patch.object(s.old, 'PUBLIC', self.public), patch.object(s.old, 'git', side_effect=git), \
                 patch.object(s, '_scan_metadata', side_effect=scan):
                with self.assertRaises(ValueError): s.build_inventory(future)
        finally:
            if junction.is_junction(): junction.rmdir()

    def test_recognized_missing_or_linked_file_does_not_silently_pass(self):
        with self.assertRaises(ValueError):
            s._scan_metadata(self.artifacts,self.public,self.new,['missing/registration.json'])
        target=self.root/'outside.json'
        target.write_text('{"test_seed":1}',encoding='utf-8')
        link=self.artifacts/'other'/'registration.json'
        link.parent.mkdir()
        try:
            link.symlink_to(target)
        except (OSError,NotImplementedError):
            self.skipTest('File symlinks are unavailable to this account')
        with self.assertRaises(ValueError):
            s._scan_metadata(self.artifacts,self.public,self.new,[])

    def test_malformed_allowed_metadata_fails_closed(self):
        source=self.artifacts/'old'/'registration.json'
        source.parent.mkdir()
        source.write_text('{"seed":1,"seed":2}',encoding='utf-8')
        with self.assertRaises(ValueError):
            s._scan_metadata(self.artifacts,self.public,self.new,[])

    @unittest.skipUnless(os.name == 'nt', 'Windows junction check')
    def test_junctions_are_rejected_without_touching_target(self):
        target=self.root/'junction-target'
        target.mkdir()
        sentinel=target/'sentinel.txt'
        sentinel.write_text('preserve',encoding='utf-8')
        junction=self.artifacts/'junction-stage'
        result=subprocess.run(['cmd','/d','/c','mklink','/J',str(junction),str(target)],
                              capture_output=True,check=False)
        if result.returncode:
            self.skipTest('Windows junction creation is unavailable')
        try:
            self.assertTrue(junction.is_junction())
            with self.assertRaises(ValueError):
                s._scan_metadata(self.artifacts,self.public,junction,[])
            with self.assertRaises(ValueError):
                s._scan_metadata(self.artifacts,self.public,self.new,[])
        finally:
            if junction.is_junction():
                junction.rmdir()
        self.assertEqual(sentinel.read_text(encoding='utf-8'),'preserve')


if __name__ == '__main__':
    unittest.main()
