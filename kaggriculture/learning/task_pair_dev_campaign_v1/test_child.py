"""Synthetic child flow; no actual files, match, model or GPU."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from . import child
from .manifest import build_manifest
from .test_manifest import inputs


ROOT = Path(__file__).resolve().parents[2]


class ChildTests(unittest.TestCase):
    def setUp(self):
        self.source = b'synthetic worker bytes'
        value = inputs(); value['worker_sha256'] = hashlib.sha256(self.source).hexdigest()
        self.manifest = build_manifest(**value)
        self.stage = ROOT / 'artifacts' / 'synthetic-not-created'
        self.plan = self.manifest.panel.plans()[0]

    def test_one_wrapper_then_private_job_and_worker(self):
        wrapper = self.stage / 'agents' / (self.plan['game'] + '.py')
        with mock.patch.object(child.worker, 'make_wrapper', return_value=wrapper), \
             mock.patch.object(child, '_read_regular', return_value=b'synthetic wrapper'), \
             mock.patch.object(child, '_write_new') as write, \
             mock.patch.object(child.worker, 'run_job') as run:
            child.run_one(manifest=self.manifest, worker_source=self.source, game_root=ROOT,
                          stage=self.stage, plan=self.plan, check=lambda: True)
        job = json.loads(write.call_args.args[1])
        self.assertEqual(job['task_checkpoint_sha256'], 'a'*64)
        self.assertEqual(job['rival_sha256'], 'e'*64)
        self.assertEqual(run.call_args.kwargs['plan'], job)
        self.assertFalse(self.stage.exists())

    def test_bad_source_or_registration_never_starts_worker(self):
        for source, check in ((b'wrong', lambda: True), (self.source, lambda: False)):
            with self.subTest(source_type=type(source)), \
                 mock.patch.object(child.worker, 'make_wrapper') as make, \
                 mock.patch.object(child.worker, 'run_job') as run:
                with self.assertRaisesRegex(RuntimeError, '^Development child failed$'):
                    child.run_one(manifest=self.manifest, worker_source=source, game_root=ROOT,
                                  stage=self.stage, plan=self.plan, check=check)
                make.assert_not_called(); run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
