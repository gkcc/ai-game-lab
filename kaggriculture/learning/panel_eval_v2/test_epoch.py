import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import epoch


class FakeContext:
    ROOT = epoch.experiment.ROOT
    prereg_commit = 'a'*40

    def __init__(self, stage):
        self.OUT = stage
        self.calls = 0
        self.after = None

    def trusted_out(self, out):
        if Path(out) != self.OUT: raise ValueError('Wrong stage')
        return self.OUT

    def deadline_check(self): pass

    def ready(self):
        self.calls += 1
        if self.after: self.after()
        return {'official_loader_sha256':'b'*64}


class EpochTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name)
        self.source = self.stage/'source'
        self.source.write_bytes(b'original')
        self.base = FakeContext(self.stage)
        self.ctx = epoch.EpochContext(self.base)
        self.cfg = {'official_loader_sha256':'b'*64}
        self.patches = [patch.object(self.ctx, '_identity_paths', side_effect=lambda:(self.cfg, frozenset([self.source]))),
                        patch.object(epoch.old, 'clean_head'), patch.object(epoch.experiment, '_check_official_runtime')]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_nested_legacy_checks_reuse_one_full_scan(self):
        self.ctx.refresh()
        self.assertEqual(self.ctx.ready(self.stage), self.cfg)
        self.assertEqual(self.ctx.verify(self.stage), self.cfg)
        self.assertEqual(self.ctx.verify_commit(self.stage), self.cfg)
        self.assertEqual(self.base.calls, 1)

    def test_changed_source_fails_and_new_epoch_rechecks_fully(self):
        self.ctx.refresh()
        self.source.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.ctx.ready()
        self.ctx.refresh()
        self.assertEqual(self.base.calls, 2)
        self.assertEqual(self.ctx.ready(), self.cfg)

    def test_change_during_full_verification_invalidates_epoch(self):
        self.ctx.refresh()
        self.base.after = lambda:self.source.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.ctx.refresh()
        with self.assertRaises(ValueError): self.ctx.ready()

    def test_epoch_cannot_be_used_from_another_process(self):
        self.ctx.refresh()
        with patch.object(epoch.os, 'getpid', return_value=os.getpid()+1):
            with self.assertRaises(ValueError): self.ctx.ready()
            with self.assertRaises(ValueError): self.ctx.refresh()


class UntrustedRegistrationTests(unittest.TestCase):
    def test_precheck_does_not_read_registration_chosen_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)/'stage'
            stage.mkdir()
            base = FakeContext(stage)
            base.ready = lambda: (_ for _ in ()).throw(ValueError('untrusted registration'))
            ctx = epoch.EpochContext(base)
            bad = 'artifacts/synthetic-prior/games/partial-result.json'
            cfg = {'merged_checkpoint':bad}
            reference = {'runtime_files':{}, 'new_source_sha256':{}}
            trusted_models = {n:Path(temp)/(n+'.npz') for n in ('merged','unweighted','crop')}
            observed = []
            with patch.object(epoch.experiment, '_read', return_value=cfg), \
                 patch.object(epoch.experiment, '_reference_freeze', return_value=reference), \
                 patch.object(epoch.experiment, 'SUPPLEMENTAL_FILES', frozenset()), \
                 patch.object(epoch.experiment, '_installed_files', return_value={}), \
                 patch.object(epoch.old, 'historical_inputs', return_value=({},trusted_models,{},{})), \
                 patch.object(ctx, '_fingerprints', side_effect=lambda paths:observed.extend(paths) or {}):
                with self.assertRaises(ValueError): ctx.refresh()
            self.assertNotIn(epoch.experiment.ROOT/bad, observed)
            self.assertTrue(all(path in observed for path in trusted_models.values()))


if __name__ == '__main__':
    unittest.main()
