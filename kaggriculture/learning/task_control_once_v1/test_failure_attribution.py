"""Exercise accepted-game failure bookkeeping using synthetic receipts only."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import entry, runner
from .test_runner import FakeDriver, ROOT, STAGE, limits, manifest_and_plan


class FailureAttributionTests(unittest.TestCase):
    def test_failed_receipt_after_acceptance_does_not_guess_zero(self):
        manifest, plan = manifest_and_plan()
        drivers = []
        completed_writes = []
        parent_receipts = []

        class TransientFailureDriver(FakeDriver):
            def failure(self, game, completed):
                completed_writes.append((game, completed))
                raise OSError('synthetic failure writing campaign receipt')

        def driver_factory(**kwargs):
            driver = TransientFailureDriver(kwargs['manifest'], kwargs['check'])
            drivers.append(driver)
            return driver

        ctx = dict(root=ROOT, stage=STAGE, manifest=manifest, manifest_raw=b'synthetic',
                   plan=plan, limits=limits(), commit='a' * 40, attempt_raw=b'synthetic',
                   modules={'driver': SimpleNamespace(LocalDriver=driver_factory)}, runner=runner)

        def stage_write(_ctx, relative, _data):
            if relative == 'control-result.local.json':
                raise OSError('synthetic result write failure')

        def record_parent_failure(value):
            parent_receipts.append(entry._failure_record(value))

        with patch.object(entry, '_build_context', return_value=ctx), \
             patch.object(entry, '_claim_attempt', side_effect=lambda value: value.update(_attempt_created=True)), \
             patch.object(entry, '_bind_private'), \
             patch.object(entry, '_check', return_value=True), \
             patch.object(entry, '_authorize'), \
             patch.object(entry, '_write_stage', side_effect=stage_write), \
             patch.object(entry, '_attempt_failure', side_effect=record_parent_failure), \
             patch.object(runner, '_verify_assets'), \
             patch.object(runner.worker, '_quiet_output', side_effect=nullcontext), \
             patch.object(runner.driver_module, '_owned_size', return_value=100), \
             patch.object(Path, 'is_dir', return_value=True):
            with self.assertRaisesRegex(RuntimeError, '^Single control stopped$'):
                entry.execute(game_root=ROOT, registration_commit='a' * 40)

        self.assertEqual(len(drivers), 1)
        self.assertEqual((drivers[0].launched, drivers[0].accepted), (1, 1))
        self.assertEqual(completed_writes, [(entry.GAME, 1)])
        self.assertEqual(len(parent_receipts), 1)
        self.assertIsNone(parent_receipts[0]['accepted_games'])
        self.assertFalse(parent_receipts[0]['retry_allowed'])


if __name__ == '__main__':
    unittest.main()
