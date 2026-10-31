"""Synthetic child wiring; no old archives, private seed values, models or jobs."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import psutil

from learning.task_pair_fresh_multi_v1 import entry
from learning.task_student_anchor_collection_v1 import sources
from learning.task_teacher_dev_deviation_v1 import membership
from learning.task_teacher_transfer_dataset_v1.test_integration import success_job
from . import archive, experiment


class ChildWiringTests(unittest.TestCase):
    def exercise(self, *, fail_second=False, attempt=1, reject_job=False):
        head, binding_sha = 'a' * 40, 'b' * 64
        job_name = 'synthetic-job'
        binding = dict(train_sha256='train', dev_sha256='dev', baseline_sha256='base',
                       schema_sha256='semantic', historical=dict(
                           schema_sha256='historical', archive_sha256={'old': 'old', 'new': 'new'}))
        claim = dict(schema='teacher-cross-source-claim-v1', commit=head, attempt=attempt,
                     freeze_sha256=binding_sha, job_token_sha256='c' * 64, retry_allowed=False)
        files = {
            'claim.local.json': claim,
            'attempt.local.json': dict(schema='teacher-cross-source-attempt-v1',
                                       commit=head, retry_allowed=False),
            'owner.local.json': {'synthetic': True},
        }
        calls = []
        fake_job = Mock(side_effect=ValueError('synthetic job mismatch') if reject_job else None)
        fake_owner = Mock()
        features = np.zeros((2, 373), dtype=np.float32)
        train = [SimpleNamespace(features=features, target_index=0) for _ in range(59)]
        dev = [SimpleNamespace(features=features, target_index=1) for _ in range(62)]
        contract = dict(registration_commit=experiment.DATA_COMMIT, train_sha256='train', dev_sha256='dev')
        context = SimpleNamespace(task_sha256='base', model_normalization=lambda: (
            np.zeros(373, dtype=np.float64), np.ones(373, dtype=np.float64), {}))

        def read(path, _limit=experiment.MAX_METADATA):
            return json.dumps(files[Path(path).name]).encode()

        def write(path, value):
            name = Path(path).name
            if name in files:
                raise FileExistsError('synthetic marker already exists')
            files[name] = value

        def scan(_path, *, expected_sha256, expected_groups, expected_candidates, guard, consume):
            calls.append(expected_sha256)
            # Callback data and labels are entirely synthetic. The failure is
            # after callback delivery, where a real second rehash would occur.
            consume(0, np.zeros(275, dtype=np.float32), np.zeros((2, 98), dtype=np.float32), 0)
            if fail_second and len(calls) == 2:
                raise ValueError('synthetic final rehash failure')
            return dict(groups=expected_groups, candidates=expected_candidates, verified_after_scan=True)

        with ExitStack() as stack:
            for context_patch in (
                patch.object(experiment, 'PRIVATE', Path('synthetic-stage')),
                patch.object(experiment, '_runtime_binding', return_value=(head, binding, binding_sha)),
                patch.object(experiment, '_read', side_effect=read),
                patch.object(experiment.previous, '_write_new', side_effect=write),
                patch.object(experiment.previous, '_guard'),
                patch.object(experiment.transfer, 'load_trainable', return_value=(train, dev, contract)),
                patch.object(sources, 'open_context', return_value=context),
                patch.object(experiment, '_schema_identity', return_value='semantic'),
                patch.object(experiment, 'GROUPS', {'old': 1, 'new': 1}),
                patch.object(experiment, 'CANDIDATES', {'old': 2, 'new': 2}),
                patch.object(archive, 'scan_archive', side_effect=scan),
                patch.object(membership, 'verify_worker', fake_job),
                patch.object(entry, 'verify_live_owner', fake_owner),
                patch.object(Path, 'exists', return_value=False),
                patch.dict(os.environ, {membership.ENV_KEY: job_name}),
            ):
                stack.enter_context(context_patch)
            try:
                experiment.child()
                error = None
            except ValueError as exc:
                error = str(exc)
        return SimpleNamespace(files=files, calls=calls, job=fake_job, owner=fake_owner,
                               head=head, binding_sha=binding_sha, claim=claim,
                               job_name=job_name, error=error)

    def test_both_scans_verified_before_private_result(self):
        result = self.exercise()
        self.assertIsNone(result.error)
        self.assertEqual(result.calls, ['old', 'new'])
        self.assertTrue(result.files['worker-result.local.json']['passed'])
        self.assertFalse(result.files['worker-result.local.json']['old_archive_seed_payload_decoded'])
        self.assertNotIn('worker-failed.local.json', result.files)
        self.assertEqual(result.claim['attempt'], 1)
        result.job.assert_called_once_with(result.job_name, result.claim, result.binding_sha)
        self.assertEqual(result.owner.call_count, 1)
        self.assertEqual(result.owner.call_args.kwargs['commit'], result.head)
        self.assertEqual(result.owner.call_args.kwargs['binding_sha'], result.binding_sha)

    def test_second_scan_final_rehash_failure_never_seals(self):
        result = self.exercise(fail_second=True)
        self.assertEqual(result.error, 'synthetic final rehash failure')
        self.assertEqual(result.calls, ['old', 'new'])
        self.assertNotIn('worker-result.local.json', result.files)
        self.assertEqual(result.files['worker-failed.local.json']['completed_archives'], 1)
        self.assertFalse(result.files['worker-failed.local.json']['retry_allowed'])

    def test_bad_attempt_or_job_membership_stops_before_scanning(self):
        invalid_attempt = self.exercise(attempt=2)
        self.assertIsNotNone(invalid_attempt.error)
        invalid_attempt.job.assert_not_called()
        self.assertEqual(invalid_attempt.calls, [])
        self.assertNotIn('child-attempt.local.json', invalid_attempt.files)
        rejected_job = self.exercise(reject_job=True)
        self.assertEqual(rejected_job.error, 'synthetic job mismatch')
        rejected_job.job.assert_called_once()
        self.assertEqual(rejected_job.calls, [])
        self.assertNotIn('child-attempt.local.json', rejected_job.files)


class ParentSealTests(unittest.TestCase):
    def exercise(self, *, fail_job=False):
        head, binding_sha = 'a' * 40, 'b' * 64
        files, calls = {}, []
        rows = ([dict(split='train', raw='none', normalized='none') for _ in range(59)]
                + [dict(split='dev', raw='none', normalized='none') for _ in range(62)])
        summary = {kind: {split: {'same': 0, 'conflict': 0, 'none': count}
                          for split, count in (('train', 59), ('dev', 62))}
                   for kind in ('raw', 'normalized')}
        result = dict(schema='teacher-cross-source-private-result-v1', registration_commit=head,
            old_sources=['old', 'new'], old_groups=dict(experiment.GROUPS),
            old_candidates=dict(experiment.CANDIDATES), new_train_groups=59, new_dev_groups=62,
            rows=rows, summary=summary, model_forward=False, training=False, new_games=0,
            old_archive_seed_payload_decoded=False, passed=True)

        def read(path, _limit=experiment.MAX_METADATA):
            name = Path(path).name
            if name not in files: raise FileNotFoundError('synthetic')
            return files[name]

        def write(path, value):
            name = Path(path).name
            if name in files: raise FileExistsError('synthetic')
            files[name] = experiment.previous._encoded(value)

        def supervised(command, _cwd, _limits, _job_name):
            calls.append(command)
            claim = experiment.previous._document(files['claim.local.json'])
            self.assertEqual(claim['attempt'], 1)
            if not fail_job:
                write('child-attempt.local.json', dict(schema='teacher-cross-source-child-attempt-v1',
                                                      commit=head, retry_allowed=False))
                write('worker-result.local.json', result)
            return {**success_job(), 'passed': not fail_job}

        with ExitStack() as stack:
            for context_patch in (
                patch.object(experiment, 'PRIVATE', Path('synthetic-stage')),
                patch.object(experiment, '_windows', return_value=True),
                patch.object(experiment.previous, 'physical_available', return_value=8 * 1024**3),
                patch.object(experiment.shutil, 'disk_usage', return_value=SimpleNamespace(free=8 * 1024**3)),
                patch.object(experiment, '_runtime_binding', return_value=(head, {}, binding_sha)),
                patch.object(experiment, '_registered_binding', return_value=({}, binding_sha)),
                patch.object(experiment, '_read', side_effect=read),
                patch.object(experiment.previous, '_write_new', side_effect=write),
                patch.object(experiment.supervisor, '_supervise_owned', side_effect=supervised),
                patch.object(entry, 'owner_marker', return_value=experiment.previous._encoded(dict(
                    schema='task-pair-fresh-multi-owner-v1', registration_commit=head,
                    binding_sha256=binding_sha, pid=1, created_us=1, deadline=1.0))),
                patch.object(psutil, 'Process', return_value=SimpleNamespace(create_time=lambda: 1.0)),
                patch.object(Path, 'exists', new=lambda path: path.name in files),
                redirect_stdout(io.StringIO()),
            ):
                stack.enter_context(context_patch)
            if fail_job:
                with self.assertRaises(RuntimeError): experiment.run()
                self.assertIn('failed.local.json', files)
                self.assertNotIn('closed.local.json', files)
                with self.assertRaises(ValueError): experiment.load_completed()
            else:
                experiment.run()
                self.assertEqual(experiment.load_completed()['summary'], summary)
                for name in ('claim.local.json', 'job.local.json', 'worker-result.local.json',
                             'owner.local.json', 'attempt.local.json'):
                    original = files[name]
                    files[name] += b' '
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        experiment.load_completed()
                    files[name] = original
            self.assertEqual(len(calls), 1)

    def test_success_is_sealed_and_tampering_rejected(self):
        self.exercise()

    def test_failed_owned_job_cannot_be_sealed(self):
        self.exercise(fail_job=True)


class HistoricalProjectionTests(unittest.TestCase):
    def test_extracts_only_named_hashes_from_fixed_training_receipts(self):
        old = b'{"array_sha256":"' + b'a' * 64 + b'","unrelated":"PRIVATE_SENTINEL"}'
        new = b'{"array_sha256":"' + b'b' * 64 + b'","schema_sha256":"' + b'c' * 64 + b'"}'
        calls = []
        def git(command, **_kwargs):
            calls.append(command[-1])
            if command[-1].endswith(experiment.HIST_OLD_REL): return old
            if command[-1].endswith(experiment.HIST_NEW_REL): return new
            raise AssertionError('Unexpected historical source')
        with patch.object(experiment.subprocess, 'check_output', side_effect=git):
            metadata = experiment._historical_metadata()
        self.assertEqual(set(metadata['archive_sha256']), {'old', 'new'})
        self.assertEqual(len(calls), 2)
        self.assertNotIn('PRIVATE_SENTINEL', repr(metadata))
        self.assertTrue(all('registration.json' not in value for value in calls))


if __name__ == '__main__':
    unittest.main()
