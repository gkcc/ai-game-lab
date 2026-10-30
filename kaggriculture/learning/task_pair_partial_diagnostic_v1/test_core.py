"""In-memory checks of a single pair; no files, games or private inputs."""
from copy import deepcopy
import json
import unittest

from learning.task_pair_dev_metrics_v1.test_consumer import record, turn
from learning.task_pair_historical_compare_v1.test_projection import make_row
from learning.task_pair_historical_compare_v1.test_request_prefix import replay
from .core import analyze


def fixture():
    rows = {arm: make_row('S02', 0, arm) for arm in ('baseline', 'control')}
    replays = {}
    for arm in rows:
        value = replay()
        for frame in value['steps']:
            frame[0]['observation']['farms'] = [dict(money=100), dict(money=100)]
        last = value['steps'][-1]
        own, other = (rows[arm]['result']['economy'][name]['final_cash'] for name in ('own', 'opponent'))
        last[0]['observation']['farms'] = [dict(money=own), dict(money=other)]
        last[0]['reward'], last[1]['reward'] = own, other
        replays[arm] = value
    replays['control']['steps'][1][0]['action']['farmer'] = ['NORTH']
    traces = {arm: [turn(step, [record(step)] if step == 0 else ()) for step in range(719)] for arm in rows}
    return rows, replays, traces


def streams(values):
    return {arm: (row for row in values[arm]) for arm in ('baseline', 'control')}


class CoreTests(unittest.TestCase):
    def test_one_pair_reports_only_bound_development_facts(self):
        rows, replays, traces = fixture()
        rows['baseline']['private_path'] = 'SECRET-PATH'
        replays['control']['private_value'] = 'SECRET-STATE'
        result = analyze(rows=rows, replays=replays, traces=streams(traces), check=lambda: True)
        self.assertEqual((result['accepted_games'], result['paired_games'], result['seed_alias']), (2, 1, 'S01'))
        self.assertEqual((result['games']['baseline']['outcome'], result['games']['control']['outcome']),
                         ('loss', 'win'))
        self.assertEqual(result['delta']['own_net_profit'], 6)
        self.assertEqual(result['delta']['opponent_net_profit'], 0)
        self.assertEqual(result['delta']['margin'], 6)
        self.assertEqual((result['complete_request_prefix']['status'], result['complete_request_prefix']['step']),
                         ('observed', 0))
        self.assertEqual(result['task_common_history']['first_joint_request_difference'], 'unknown')
        for arm in ('baseline', 'control'):
            game = result['games'][arm]
            self.assertEqual(game['selected_task_counts']['PLANT'], 1)
            self.assertEqual(game['submitted_unit_request_counts']['movement'], 1)
            self.assertEqual(game['submitted_unit_request_counts']['other'], 1)
        self.assertEqual(result['decision_distribution_scope'],
                         'selected_tasks_full_game_different_states_descriptive')
        self.assertEqual((result['new_games'], result['new_official_transitions'], result['promotion']), (0, 0, False))
        rendered = json.dumps(result)
        for secret in ('SECRET-PATH', 'SECRET-STATE', str(rows['baseline']['seed']),
                       rows['baseline']['task_checkpoint_sha256'], 'NORTH', 'PRIVATE-REQUEST'):
            self.assertNotIn(secret, rendered)

    def test_unmatched_economy_or_pair_identity_fails(self):
        rows, replays, traces = fixture()
        variants = []
        bad = deepcopy(rows); bad['control']['seed'] += 1; variants.append((bad, replays))
        bad = deepcopy(rows); bad['control']['arm'] = 'baseline'; variants.append((bad, replays))
        bad = deepcopy(rows); bad['control']['side'] = True; variants.append((bad, replays))
        bad = deepcopy(rows); bad['control']['crop_checkpoint_sha256'] = 'a' * 64; variants.append((bad, replays))
        bad = deepcopy(rows); bad['control']['result']['score'] = .5; variants.append((bad, replays))
        bad = deepcopy(rows)
        for arm in ('baseline', 'control'):
            for side in ('own', 'opponent'):
                bad[arm]['result']['economy'][side]['initial_cash'] += 20
                bad[arm]['result']['economy'][side]['final_cash'] += 20
        variants.append((bad, replays))
        bad = deepcopy(replays); bad['control']['steps'][-1][0]['reward'] += 1; variants.append((rows, bad))
        bad = deepcopy(replays); bad['control']['steps'][-1][0]['observation']['farms'][0]['money'] += 1; variants.append((rows, bad))
        for pair_rows, pair_replays in variants:
            with self.assertRaises((ValueError, KeyError)):
                analyze(rows=pair_rows, replays=pair_replays, traces=streams(traces), check=lambda: True)

    def test_trace_must_complete_and_resource_guard_must_remain_true(self):
        rows, replays, traces = fixture()
        broken = deepcopy(traces); broken['control'].pop()
        with self.assertRaises(ValueError):
            analyze(rows=rows, replays=replays, traces=streams(broken), check=lambda: True)
        count = [0]
        def guard():
            count[0] += 1
            return count[0] < 10
        with self.assertRaises(ValueError):
            analyze(rows=rows, replays=replays, traces=streams(traces), check=guard)
        self.assertEqual(count[0], 10)


if __name__ == '__main__':
    unittest.main()
