"""Focused entry and collector regressions without models, devices, or games."""
import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

from learning.task_completion_bc_v1 import numeric
from . import campaign, collect, gpu_entry, training


class WorkerTest(unittest.TestCase):
    def test_progress_and_primary_failure_are_preserved(self):
        records = {}
        def fail(out, progress):
            progress(dict(phase='update_started', updates_started=1))
            raise ValueError('synthetic failure')
        def exists(path):
            return path.name == 'fit-started.json'
        with patch.object(gpu_entry, '_worker_impl', side_effect=fail), \
             patch.object(Path, 'open', mock_open()), \
             patch.object(Path, 'exists', autospec=True, side_effect=exists), \
             patch.object(gpu_entry, 'write_new', side_effect=lambda p,v: records.update({p.name:v})):
            with self.assertRaises(ValueError):
                gpu_entry.worker(Path('synthetic'))
        failure = records['fit-worker-failure.json']
        self.assertEqual(failure['error_kind'], 'ValueError')
        self.assertEqual(failure['last_progress']['phase'], 'update_started')
        self.assertEqual(failure['last_progress']['updates_started'], 1)
        self.assertEqual(failure['last_progress']['updates_completed'], 0)
        self.assertFalse(failure['retry_allowed'])

    def test_old_cache_identity_is_checked_again_after_decode(self):
        cfg = {'data_limits': {'max_groups_per_split': 4, 'max_candidates_per_split': 20},
               'train_seeds': [7], 'training': {'training_seed': 9}}
        source = {'passed': True, 'array_sha256': 'expected'}
        def read(path):
            return {'fit-started.json': {'train_data_sha256': 'marker'},
                    'train-data-result.json': source}[path.name]
        def sha(path):
            if path.name == 'train-data-result.json': return 'marker'
            return next(hashes)
        hashes = iter(['expected', 'changed'])
        progress = []
        with patch.object(gpu_entry, 'ready', return_value=cfg), \
             patch.object(gpu_entry, 'read', side_effect=read), \
             patch.object(gpu_entry, 'sha', side_effect=sha), \
             patch.object(numeric, 'read', return_value='synthetic') as load, \
             patch.object(training, 'fit') as fit:
            with self.assertRaisesRegex(ValueError, 'during reading'):
                gpu_entry._worker_impl(Path('synthetic'), progress.append)
        self.assertEqual(load.call_args.args[0], gpu_entry.PREVIOUS/'private/train.npz')
        self.assertEqual(progress[-1]['phase'], 'decoding_training_data')
        fit.assert_not_called()


class CollectEntryTest(unittest.TestCase):
    def test_direct_entry_checks_seal_and_restores_bindings_on_failure(self):
        original = tuple(getattr(collect.core, n) for n in ('ready', 'write_new', 'deadline_check'))
        def fail(_):
            self.assertIs(collect.core.ready, collect.experiment.ready)
            raise ValueError('synthetic')
        with patch.object(campaign, 'verify_model_seal') as seal, \
             patch.object(collect.core, 'test', side_effect=fail):
            with self.assertRaises(ValueError): collect.test('synthetic')
        seal.assert_called_once_with(Path('synthetic').resolve())
        self.assertEqual(tuple(getattr(collect.core, n) for n in ('ready', 'write_new', 'deadline_check')), original)

    def test_bad_seal_prevents_collection(self):
        with patch.object(campaign, 'verify_model_seal', side_effect=ValueError), \
             patch.object(collect.core, 'test') as core:
            with self.assertRaises(ValueError): collect.test('synthetic')
        core.assert_not_called()


class IndependentBindingTest(unittest.TestCase):
    def test_altered_method_cannot_authorize_evaluation(self):
        cfg = {'test_seeds': [7], 'independent_sample': 'fixed',
               'independent_limits': {'compressed_bytes': 100, 'decoded_bytes': 100}}
        replay = {'passed': True, 'game': 'holdout-7-round2-0-teacher_task',
                  'official_transitions': 719, 'sample_count': 20, 'matched': 20,
                  'selection_rule': 'fixed', 'raw_reads': {'files_opened': 2, 'files_completed': 2,
                  'raw_decode_passes_per_file': 1, 'compressed_input_read_bytes': 3, 'decoded_input_bytes': 4}}
        review = {'passed': True, 'official_transitions': 719, 'sample_count': 20, 'sample_matched': 20,
                  'freeze_sha256': 'freeze.json', 'model_seal_sha256': 'model-seal.json',
                  'games_result_sha256': 'games-result.json', 'game_ledger_sha256': 'game-ledger.json',
                  'game_audit_ledger_sha256': 'game-audit-ledger.json',
                  'test_data_result_sha256': 'test-data-result.json',
                  'test_label_ledger_sha256': 'test-label-ledger.json',
                  'verification_sha256': 'independent-replay-verification.json',
                  'method_sha256': 'different-from-source'}
        def read(path):
            return review if path.name == 'independent-result-review.json' else replay
        with patch.object(campaign, 'read', side_effect=read), \
             patch.object(campaign, 'sha', side_effect=lambda p: Path(p).name):
            with self.assertRaises(ValueError):
                campaign.verify_independent_review(Path('synthetic'), cfg)
            review['method_sha256'] = 'independent_method.py'
            self.assertIs(campaign.verify_independent_review(Path('synthetic'), cfg), review)


if __name__ == '__main__':
    unittest.main()
