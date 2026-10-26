"""Pipe-scope and original-error regressions without real models or processes."""
from copy import deepcopy
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_anchor_once_v1.test_storage import _scratch
from learning.task_anchor_once_v1.experiment import STORAGE
from . import training,campaign


class ReentryTests(unittest.TestCase):
    def test_frozen_worker_launch_has_writable_ipc_and_adapter_shares_parent_cap(self):
        cfg=dict(training=deepcopy(training.method.TRAINING),training_seed=2026102707,
            storage=deepcopy(STORAGE))
        with _scratch() as folder:
            out=folder/'output';(out/'model').mkdir(parents=True)
            # An opaque synthetic byte fixture is hashed, never decoded as a model.
            (out/'model/trained.npz').write_bytes(b'synthetic opaque identity fixture')
            digest=training.sha(out/'model/trained.npz')
            training.write_new(out/'registration.json',cfg)
            training.write_new(out/'fit-result.json',dict(completed=True,optimizer_updates=512,checkpoint_sha256=digest))
            training.write_new(out/'offline-gate.json',dict(checkpoint_sha256=digest,passed=False))
            def pipe_launcher(parent,target,args,**limits):
                self.assertIs(target,training.method._worker)
                self.assertEqual(args[:2],(str(out.resolve()),cfg))
                read_fd,write_fd=os.pipe()
                try:
                    # This actual operation fails under the defective old guard.
                    with open(write_fd,'wb',closefd=False) as stream:stream.write(b'x')
                    self.assertEqual(os.read(read_fd,1),b'x')
                finally:os.close(read_fd);os.close(write_fd)
                training.write_new(parent/'launch-started.json',{'synthetic':True})
                training.write_new(parent/'launch-result.json',{'synthetic':True})
                return dict(parent_receipt_bytes_total=sum(p.stat().st_size for p in parent.iterdir()),
                    parent_receipt_refused=False)
            with patch.object(training,'training_ready',return_value=cfg),patch.object(training,'launch',side_effect=pipe_launcher):
                result=training.fit(out)
            self.assertEqual(result['checkpoint_sha256'],digest)
            adapter=training.read(out/'fit-adapter-result.json')
            self.assertFalse(adapter['old_globally_guarded_launcher_called'])
            parent=sum(p.stat().st_size for p in (out/'training-parent').iterdir())
            self.assertEqual(adapter['parent_launch_write_bytes'],parent)
            self.assertEqual(adapter['adapter_write_allowance_bytes']+parent,65536)
            self.assertLessEqual(parent+(out/'fit-adapter-result.json').stat().st_size,65536)
        self.assertFalse(folder.exists())

    def test_gpu_dispatch_keeps_original_error_when_cleanup_also_fails(self):
        def wait(**kwargs):raise RuntimeError('secondary transport error')
        def join(**kwargs):raise OSError('secondary reader error')
        child=SimpleNamespace(process=SimpleNamespace(pid=123,poll=lambda:None,wait=wait),
            thread=SimpleNamespace(join=join,is_alive=lambda:True))
        cfg=dict(storage=deepcopy(STORAGE),gpu_guard={'process_seconds':1200})
        with _scratch() as folder:
            out=folder/'output';out.mkdir()
            for name in ('registration.json','freeze.json','pre-execution-commit.json','reused-cache-inputs.json'):
                training.write_new(out/name,{'synthetic':True})
            training.write_new(out/'gpu-launch.local.json',dict(distro='synthetic',workdir='synthetic',python='synthetic'))
            runner=campaign.Runner(out,cfg,None)
            with patch.object(campaign,'ROOT',folder),patch.object(campaign,'ready'), \
                 patch.object(runner,'storage_ready'),patch.object(campaign,'CapturedProcess',return_value=child), \
                 patch.object(runner,'limits',side_effect=TimeoutError('original campaign limit')):
                with self.assertRaisesRegex(TimeoutError,'original campaign limit'):runner.gpu_fit()
            self.assertFalse(runner.gpu_cleanup['passed'])
            self.assertIn('transport:RuntimeError',runner.gpu_cleanup['errors'])
            self.assertIn('reader:OSError',runner.gpu_cleanup['errors'])
            self.assertIsNone(runner.gpu_cleanup['linux_cleanup_evidence'])
            self.assertFalse((out/'training-process-result.json').exists())
        self.assertFalse(folder.exists())


if __name__=='__main__':unittest.main()
