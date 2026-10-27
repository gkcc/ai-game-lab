"""Small regressions for identity and process-cleanup contracts; no real data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from . import campaign, experiment


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class BoundEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.out = Path('synthetic-not-created')
        self.store = {'freeze.json': {'version': 1}, 'model-final.npz': {'weights': 1},
                      'fit-transport-result.json': {'passed': True},
                      'train-data-result.json': {'passed': True}}
        fit = dict(passed=True, updates=128, checkpoint_sha256=identity(self.store['model-final.npz']),
                   train_data_sha256=identity(self.store['train-data-result.json']), baseline={'WATER': 10},
                   training_summary=dict(synchronized_updates=128, gpu_verified=True, weights_changed=True,
                       nonzero_gradient_updates=128,
                       inference_parity=dict(groups=16, candidate_rows=20, synchronized=True)))
        self.store['fit-result.json'] = fit
        self.store['model-seal.json'] = dict(passed=True, freeze_sha256=identity(self.store['freeze.json']),
            fit_result_sha256=identity(fit), fit_transport_sha256=identity(self.store['fit-transport-result.json']),
            checkpoint_sha256=identity(self.store['model-final.npz']))

    def fake_read(self, path):
        return self.store[Path(path).name]

    def fake_sha(self, path):
        return identity(self.store[Path(path).name])

    def test_modified_baseline_breaks_model_seal(self):
        with patch.object(campaign, 'read', self.fake_read), patch.object(campaign, 'sha', self.fake_sha):
            campaign.verify_model_seal(self.out)
            self.store['fit-result.json']['baseline']['WATER'] = 99
            with self.assertRaises(ValueError):
                campaign.verify_model_seal(self.out)

    def test_review_must_bind_exact_result_and_registered_sample(self):
        cfg = dict(test_seeds=[42, 43], independent_sample='fixed rule',
                   independent_limits=dict(compressed_bytes=1024, decoded_bytes=4096))
        for name in ('games-result.json','game-ledger.json','game-audit-ledger.json',
                     'test-data-result.json','test-label-ledger.json'):
            self.store[name] = {'name': name}
        self.store['independent-replay-verification.json'] = dict(passed=True,
            game='holdout-42-round2-0-teacher_task', official_transitions=719,
            sample_count=20, matched=20, selection_rule='fixed rule',
            raw_reads=dict(files_opened=2, files_completed=2, raw_decode_passes_per_file=1,
                           compressed_input_read_bytes=100, decoded_input_bytes=200))
        files = {'freeze_sha256':'freeze.json', 'model_seal_sha256':'model-seal.json',
                 'games_result_sha256':'games-result.json', 'game_ledger_sha256':'game-ledger.json',
                 'game_audit_ledger_sha256':'game-audit-ledger.json', 'test_data_result_sha256':'test-data-result.json',
                 'test_label_ledger_sha256':'test-label-ledger.json',
                 'verification_sha256':'independent-replay-verification.json'}
        review = dict(passed=True, official_transitions=719, sample_count=20, sample_matched=20,
                      **{k:identity(self.store[v]) for k,v in files.items()})
        self.store['independent-result-review.json'] = review
        with patch.object(campaign, 'read', self.fake_read), patch.object(campaign, 'sha', self.fake_sha):
            campaign.verify_independent_review(self.out, cfg)
            self.store['game-ledger.json']['tampered'] = True
            with self.assertRaises(ValueError):
                campaign.verify_independent_review(self.out, cfg)
            self.store['game-ledger.json'].pop('tampered')
            self.store['independent-replay-verification.json']['matched'] = 19
            review['verification_sha256'] = identity(self.store['independent-replay-verification.json'])
            with self.assertRaises(ValueError):
                campaign.verify_independent_review(self.out, cfg)

    def test_cleanup_failure_is_not_accepted(self):
        ok = dict(owned_instances_remaining=0, reader_joined=True, launched_process_reaped=True)
        campaign.require_cleanup(ok)
        for key, value in [('owned_instances_remaining',1), ('reader_joined',False), ('launched_process_reaped',False)]:
            changed = dict(ok, **{key:value})
            with self.assertRaises(RuntimeError):
                campaign.require_cleanup(changed)

    def test_historical_labels_cannot_be_replaced_with_manifest(self):
        game = dict(key='g', side=0, replay={'sha256':'r'}, trace={'sha256':'t'})
        part = dict(sha256='accepted', rows=10)
        report = dict(game='g', side=0, labels_sha256='accepted', rows=10, replay_sha256='r', trace_sha256='t')
        experiment.validate_historical_report(game, part, report)
        replacement = dict(part, sha256='different')
        with self.assertRaises(ValueError):
            experiment.validate_historical_report(game, replacement, report)


if __name__ == '__main__':
    unittest.main()
