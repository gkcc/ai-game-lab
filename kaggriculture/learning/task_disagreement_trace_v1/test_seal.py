"""Synthetic in-memory seal checks; registration identity is stubbed.

No files, historic traces, checkpoints, private seed values, or Jobs are
opened. This exercises the real completion consumer after a fixed binding,
not the authenticity of the repository registration itself.
"""
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

from learning.task_teacher_transfer_dataset_v1.test_integration import success_job
from . import experiment as exp
from .test_experiment import META, MEAN, SCALE, full_examples, guard


class SealTests(unittest.TestCase):
    def test_completed_dataset_loads_and_each_receipt_tampering_is_rejected(self):
        head, binding_sha = 'a' * 40, 'b' * 64
        train, diagnostic, summary = exp._build_dataset(
            full_examples(), mean=MEAN, scale=SCALE, metadata=META, guard=guard)
        self.assertEqual(summary['data_gate'], 'passed')
        files = {
            'train.local.npz': exp._split_bytes(train, guard),
            'diagnostic.local.npz': exp._split_bytes(diagnostic, guard),
        }
        result = dict(schema='disagreement-trace-worker-v1', registration_commit=head,
            summary=summary, source_games=4, trace_turns=4 * 719,
            train_sha256=exp._sha(files['train.local.npz']),
            diagnostic_sha256=exp._sha(files['diagnostic.local.npz']),
            training=False, model_forward=False, new_games=0)

        def put(name, value):
            files[name] = exp.previous._encoded(value)

        put('worker-result.local.json', result)
        put('dataset-contract.local.json', exp._dataset_contract(head, result))
        put('job.local.json', success_job())
        put('attempt.local.json', dict(schema='disagreement-trace-attempt-v1',
                                      commit=head, retry_allowed=False))
        put('child-attempt.local.json', dict(schema='disagreement-trace-child-attempt-v1',
                                            commit=head, retry_allowed=False))
        put('claim.local.json', dict(schema='disagreement-trace-claim-v1', commit=head,
                                    attempt=1, freeze_sha256=binding_sha,
                                    retry_allowed=False, job_token_sha256='c' * 64))
        put('owner.local.json', dict(schema='task-pair-fresh-multi-owner-v1',
                                    registration_commit=head, binding_sha256=binding_sha,
                                    pid=1, created_us=1, deadline=1.0))
        names = ('worker-result', 'dataset-contract', 'job', 'attempt', 'owner',
                 'claim', 'child-attempt')
        put('closed.local.json', dict(schema='disagreement-trace-closed-v1',
            registration_commit=head, binding_sha256=binding_sha,
            receipts={name: exp._sha(files[name + '.local.json']) for name in names},
            retry_allowed=False))

        def read(path, limit=1024**2):
            raw = files[Path(path).name]
            if not raw or len(raw) > limit:
                raise ValueError('Synthetic read budget exceeded')
            return raw

        tamper = ('train.local.npz', 'diagnostic.local.npz',
                  'worker-result.local.json', 'dataset-contract.local.json', 'job.local.json',
                  'attempt.local.json', 'owner.local.json', 'claim.local.json',
                  'child-attempt.local.json')
        with ExitStack() as stack:
            for p in (
                patch.object(exp, 'PRIVATE', Path('synthetic-stage')),
                patch.object(exp, '_registered_binding',
                             return_value=({'operation_columns': list(range(5))}, binding_sha)),
                patch.object(exp, '_read', side_effect=read),
                patch.object(Path, 'exists', return_value=False),
            ):
                stack.enter_context(p)
            self.assertEqual(exp.load_completed()['summary'], summary)
            for name in tamper:
                original = files[name]
                try:
                    files[name] = original + b' '
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        exp.load_completed()
                finally:
                    files[name] = original


if __name__ == '__main__':
    unittest.main()
