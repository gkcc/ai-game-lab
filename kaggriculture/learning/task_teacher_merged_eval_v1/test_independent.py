import io
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from learning.task_choice_v1.features import OPS
from . import campaign, experiment, independent_method as method, independent_runner as runner
from .evaluation import plans


class IndependentMethodTests(unittest.TestCase):
    def test_selected_ledger_and_receipt_binding(self):
        p=plans([101,202,303,404])[0]
        row=dict(p,passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
            bank=11,opponent_bank=10,initial_bank=3,opponent_initial_bank=3,net_profit=8,
            opponent_net_profit=7,margin=1,score=1.)
        summary=campaign.sanitized(row)
        audit=dict(game=p['game'],receipt_sha256=dict(row='a'*64,audit='b'*64))
        method.bind_selected(summary,row,audit,'a'*64,'b'*64)
        for changed in ({**audit,'game':'other'},{**audit,'receipt_sha256':{'row':'c'*64,'audit':'b'*64}}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                method.bind_selected(summary,row,changed,'a'*64,'b'*64)
        with self.assertRaises(ValueError): method.bind_selected({**summary,'bank':12},row,audit,'a'*64,'b'*64)

    def test_fixed_callback_sample_and_operation_counts(self):
        identities=[('holdout-S01-round2-0-merged',i,0) for i in range(30)]
        chosen=method.sample_identities(identities)
        self.assertEqual(chosen,method.sample_identities(identities[::-1])); self.assertEqual(len(chosen),20)
        counts=method.sample_operation_counts(chosen,{identity:OPS[0] for identity in chosen})
        self.assertEqual(sum(counts.values()),20); self.assertEqual(counts[OPS[0]],20)
        with self.assertRaises(ValueError): method.sample_identities(identities+[identities[0]])
        with self.assertRaises(ValueError): method.sample_identities(identities,True)
        with self.assertRaises(ValueError): method.sample_operation_counts(chosen,{identity:'unknown' for identity in chosen})

    def test_invalid_runner_marker_prevents_replay(self):
        out=Path('synthetic-output').resolve()
        with patch.object(method,'OUT',out),patch.object(method,'trusted_out',return_value=out),patch.object(method,'ready',return_value={}), \
             patch.object(method,'verify_commit'),patch.object(method,'read',return_value={'attempt':2}), \
             patch.object(method,'sha',return_value='a'*64),patch.object(method,'create_marker') as marker:
            with self.assertRaises(ValueError): method._run(out)
            marker.assert_not_called()

    def test_python_output_is_not_exposed(self):
        def noisy(_):
            import sys
            print('private value'); print('private value',file=sys.stderr)
            return {'passed':True}
        stdout,stderr=io.StringIO(),io.StringIO()
        with patch.object(method,'_run',noisy),redirect_stdout(stdout),redirect_stderr(stderr):
            self.assertEqual(method.run(),{'passed':True})
        self.assertEqual(stdout.getvalue(),''); self.assertEqual(stderr.getvalue(),'')

    def test_stage_link_rejected_by_all_marker_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project'; artifacts=root/'artifacts'; artifacts.mkdir(parents=True)
            outside=Path(tmp)/'outside'; outside.mkdir()
            sentinel=outside/'sentinel'; sentinel.write_bytes(b'preserve')
            stage=artifacts/experiment.STAGE_NAME
            if os.name=='nt':
                try:
                    subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(stage),str(outside)],
                                   check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                except (OSError,subprocess.CalledProcessError) as exc:
                    self.skipTest('Directory junction unavailable: '+type(exc).__name__)
            else:
                try: stage.symlink_to(outside,target_is_directory=True)
                except (OSError,NotImplementedError) as exc: self.skipTest('Directory link unavailable: '+type(exc).__name__)
            try:
                state=dict(created=False,complete=False)
                with patch.object(experiment,'ROOT',root),patch.object(experiment,'OUT',stage), \
                     patch.object(method,'OUT',stage),patch.object(runner,'OUT',stage):
                    for action in (lambda:method.create_marker(stage,state),lambda:method._run(stage),
                                   lambda:runner.run(stage),lambda:runner.finalize(stage)):
                        with self.assertRaises(ValueError): action()
                self.assertEqual(state,dict(created=False,complete=False))
                self.assertEqual(sentinel.read_bytes(),b'preserve')
                self.assertEqual({p.name for p in outside.iterdir()},{'sentinel'})
            finally:
                stage.rmdir() if os.name=='nt' else stage.unlink()


if __name__=='__main__': unittest.main()
