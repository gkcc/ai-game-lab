"""Synthetic serial coordination; no files, matches, models or GPU."""
from copy import deepcopy
import unittest
from unittest import mock

from . import campaign
from .manifest import build_manifest
from .test_manifest import inputs
from .test_acceptance import supervision
from learning.task_pair_dev_games_v1 import capture


def limits():
    reserve = 139 * 1024**2
    return dict(campaign_wall_seconds=3600, initial_available_bytes=4*1024**3,
                between_available_bytes=2*1024**3, disk_free_floor_bytes=4*1024**3,
                total_output_bytes=24*reserve+18*1024**2, per_game_reserve_bytes=reserve,
                replay_decoded_bytes=campaign.REPLAY_DECODED_BYTES,
                trace_compressed_bytes=capture.MAX_COMPRESSED_BYTES,
                trace_line_bytes=capture.MAX_LINE_BYTES, trace_decoded_bytes=capture.MAX_DECODED_BYTES,
                metrics_bytes=capture.MAX_METRICS_BYTES,
                job=dict(max_rss_bytes=2*1024**3, max_commit_bytes=3*1024**3,
                         min_available_bytes=2*1024**3, wall_seconds=20,
                         poll_seconds=.25, cleanup_seconds=10))


class FakeDriver:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at, self.finished, self.failures = [], fail_at, False, []
    def preflight(self, plan, fixed, remaining):
        return True
    def claim(self):
        self.calls.append(('claim', None)); return True
    def preflight_comparison(self, fixed, remaining):
        return True
    def launch(self, plan, job):
        self.calls.append(('launch', plan['game']))
        return dict(supervision(), job_members_zero=False) if len([x for x in self.calls if x[0]=='launch']) == self.fail_at else supervision()
    def read_bundle(self, plan):
        self.calls.append(('read', plan['game'])); return {}
    def record_supervision(self, plan, result):
        self.calls.append(('supervision', plan['game'])); return True
    def verify_files(self, *_):
        return True
    def record_acceptance(self, plan, bundle):
        self.calls.append(('acceptance', plan['game'])); return True
    def compare(self, left, right):
        self.calls.append(('compare', left['game']))
        return dict(reference=left['arm'], candidate=right['arm'], seed=left['seed'],
                    seed_alias=left['seed_alias'], opponent=left['opponent'], side=left['side'],
                    result=dict(comparison_complete=True, matched_calls=0,
                                matched_multi_unit_opportunities=0))
    def complete(self, rows, summaries):
        self.finished = True; self.calls.append(('complete', len(rows))); return True
    def failure(self, game, count):
        self.failures.append((game, count))


def fake_accept(_panel, plan, _bundle, _supervision, _verify):
    return dict(game=plan['game'], seed_alias=plan['seed_alias'], seed=plan['seed'],
                opponent=plan['opponent'], side=plan['side'], arm=plan['policy'], result={})


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.manifest = build_manifest(**inputs())

    def test_all_24_serial_and_16_comparisons_after_games(self):
        driver = FakeDriver()
        with mock.patch.object(campaign, 'accept', side_effect=fake_accept), \
             mock.patch.object(campaign, 'paired', return_value={'support':'synthetic'}) as paired:
            result = campaign.run(manifest=self.manifest, limits=limits(), driver=driver,
                                  check=lambda: True, clock=lambda: 0.)
        self.assertEqual(result['games'], 24)
        self.assertFalse(result['independent_improvement'])
        names = [name for name, _ in driver.calls]
        self.assertEqual(names.count('launch'), 24)
        self.assertEqual(names.count('read'), 24)
        self.assertEqual(names.count('compare'), 16)
        self.assertLess(max(i for i, n in enumerate(names) if n == 'read'), names.index('compare'))
        self.assertEqual(paired.call_count, 2)
        self.assertTrue(driver.finished)

    def test_failed_job_stops_before_read_and_never_retries(self):
        driver = FakeDriver(fail_at=4)
        with mock.patch.object(campaign, 'accept', side_effect=fake_accept):
            with self.assertRaisesRegex(RuntimeError, '^Development campaign stopped$'):
                campaign.run(manifest=self.manifest, limits=limits(), driver=driver,
                             check=lambda: True, clock=lambda: 0.)
        self.assertEqual(sum(name == 'launch' for name, _ in driver.calls), 4)
        self.assertEqual(sum(name == 'read' for name, _ in driver.calls), 3)
        self.assertEqual(driver.failures[0][1], 3)
        self.assertFalse(driver.finished)

    def test_registration_or_preflight_failure_does_not_claim(self):
        driver = FakeDriver()
        with self.assertRaises(RuntimeError):
            campaign.run(manifest=self.manifest, limits=limits(), driver=driver,
                         check=lambda: False, clock=lambda: 0.)
        self.assertEqual(driver.calls, [])
        driver.preflight = lambda *_: False
        with self.assertRaises(RuntimeError):
            campaign.run(manifest=self.manifest, limits=limits(), driver=driver,
                         check=lambda: True, clock=lambda: 0.)
        self.assertEqual(driver.calls, [])

    def test_registered_budget_and_clock_fail_closed(self):
        for field in ('campaign_wall_seconds', 'total_output_bytes', 'per_game_reserve_bytes'):
            bad = deepcopy(limits()); bad[field] = 0
            with self.subTest(field=field), self.assertRaises(ValueError):
                campaign.limits_snapshot(bad)
        samples = iter((1., 0.))
        budget = campaign._Clock(10, lambda: next(samples))
        with self.assertRaises(TimeoutError): budget.remaining()

    def test_nul_creator_adapter_preserves_owned_handles(self):
        class Base: pass
        calls = []
        def creator(_self, command, cwd):
            calls.append((command, cwd)); return ('process', 'thread', 123)
        api = campaign.quiet_api_type(Base, creator)()
        self.assertEqual(api.create_suspended(['synthetic'], 'synthetic'), ('process', 'thread'))
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
