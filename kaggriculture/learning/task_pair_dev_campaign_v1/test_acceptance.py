"""Synthetic private receipt checks; no game, model or file access."""
import hashlib
import json
import unittest
from unittest import mock

from . import acceptance
from .manifest import build_manifest
from .test_manifest import inputs
from learning.task_pair_dev_games_v1 import capture, worker


def raw(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def supervision():
    return dict(passed=True, assigned=True, resumed=True, launcher_reaped=True,
                job_members_zero=True, error_kind=None, cleanup_error_kind=None,
                stop_reason=None, acceptance_error_kind=None, exit_code=0, rss_complete_samples=1)


def fixture():
    manifest = build_manifest(**inputs())
    plan = manifest.panel.plans()[0]
    job = dict(plan, bot_sha256='9' * 64, rival_sha256='e' * 64,
               task_checkpoint_sha256='a' * 64)
    row = dict(job, passed=True, states=720, action_turns=719, status=['DONE', 'DONE'], agent_errors=0,
               engine_sha256=acceptance.ENGINE_SHA, crop_checkpoint_sha256='d' * 64,
               banks=[15, 25], initial_banks=[5, 5], rewards=[15, 25], bank=15, opponent_bank=25,
               initial_bank=5, opponent_initial_bank=5, net_profit=10, opponent_net_profit=20,
               margin=-10, score=0., trace_sha256='5' * 64, replay_gzip_sha256='6' * 64,
               semantic_replay_sha256='7' * 64, semantic_replay_sha256_v2='8' * 64)
    official = dict(passed=True, official_transitions_reexecuted=719,
                    both_public_and_private_states_match_each_step=True,
                    inventory_order_checked_each_step=True, engine_sha256=acceptance.ENGINE_SHA,
                    semantic_sha256='7' * 64, semantic_sha256_v2='8' * 64, final_rewards=[15, 25])
    counts = dict(task_forwards=3, controller_calls=719)
    audit = dict(passed=True, game=plan['game'], errors=0, actor_actions_reproduced=719,
                 rival_actions_reproduced=719, trace_turns_reproduced=719, trace_sha256='5' * 64,
                 replay_sha256='6' * 64, official=official, primary_model_counts=counts,
                 live_model_counts=counts)
    row_raw, audit_raw = raw(row), raw(audit)
    complete = dict(passed=True, game=plan['game'], row_sha256=hashlib.sha256(row_raw).hexdigest(),
                    certificate_sha256=hashlib.sha256(audit_raw).hexdigest())
    storage = dict(passed=True, game=plan['game'], error_kind=None,
                   native_output_accounted_by=worker.NATIVE_OUTPUT_SCOPE,
                   output=dict(limit_bytes=128*1024**2, used_bytes=42,
                               remaining_bytes=128*1024**2-42, refused=0))
    own = dict(cash_reconciled=True, initial_cash=5, final_cash=15, net_profit=10)
    opponent = dict(cash_reconciled=True, initial_cash=5, final_cash=25, net_profit=20)
    metrics = dict(schema='task-pair-dev-game-metrics-v1', game=plan['game'], seed_alias=plan['seed_alias'],
                   seed=plan['seed'], opponent=plan['opponent'], side=plan['side'], arm=plan['policy'],
                   task_checkpoint_sha256='a'*64, crop_checkpoint_sha256='d'*64, trace_sha256='5'*64,
                   replay_sha256='6'*64, result=dict(actor=0, official_turns=719,
                       economy=dict(own=own, opponent=opponent), margin=-10, score=0.))
    metric_raw = raw(metrics)
    receipt = dict(schema='task-pair-dev-game-metrics-receipt-v1', game=plan['game'],
                   official_replay_passes=1, metrics_bytes=len(metric_raw),
                   metrics_sha256=hashlib.sha256(metric_raw).hexdigest(), budget_bytes=capture.MAX_METRICS_BYTES)
    bundle = dict(row=row_raw, audit=audit_raw, complete=raw(complete), storage=raw(storage),
                  metrics=metric_raw, metrics_receipt=raw(receipt))
    return manifest, plan, bundle


class AcceptanceTests(unittest.TestCase):
    def test_complete_linked_bundle_requires_file_identity(self):
        manifest, plan, bundle = fixture()
        verify = mock.Mock(return_value=True)
        result = acceptance.accept(manifest.panel, plan, bundle, supervision(), verify)
        self.assertEqual(result['result']['economy']['own']['net_profit'], 10)
        verify.assert_called_once()
        with self.assertRaises(ValueError):
            acceptance.accept(manifest.panel, plan, bundle, supervision(), lambda *_: False)

    def test_unconfirmed_job_stops_before_file_callback(self):
        manifest, plan, bundle = fixture(); verify = mock.Mock(return_value=True)
        bad = dict(supervision(), job_members_zero=False)
        with self.assertRaises(ValueError): acceptance.accept(manifest.panel, plan, bundle, bad, verify)
        verify.assert_not_called()

    def test_corrupt_receipt_and_wrong_output_scope_fail(self):
        manifest, plan, bundle = fixture()
        for key in ('row', 'audit', 'metrics'):
            changed = dict(bundle); changed[key] += b' '
            with self.subTest(receipt=key), self.assertRaises(ValueError):
                acceptance.accept(manifest.panel, plan, changed, supervision(), lambda *_: True)
        changed = dict(bundle); storage = json.loads(changed['storage'])
        storage['native_output_accounted_by'] = 'bounded parent process log'
        changed['storage'] = raw(storage)
        with self.assertRaises(ValueError):
            acceptance.accept(manifest.panel, plan, changed, supervision(), lambda *_: True)

    def test_missing_and_duplicate_private_keys_fail(self):
        manifest, plan, bundle = fixture()
        del bundle['metrics_receipt']
        with self.assertRaises(ValueError): acceptance.accept(manifest.panel, plan, bundle, supervision(), lambda *_: True)
        manifest, plan, bundle = fixture(); bundle['metrics_receipt'] = b'{"game":"x","game":"y"}'
        with self.assertRaises(ValueError): acceptance.accept(manifest.panel, plan, bundle, supervision(), lambda *_: True)


if __name__ == '__main__':
    unittest.main()
