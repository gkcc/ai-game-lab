"""Focused synthetic regressions for two consumed seed roles."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.task_prefix_diagnostic_v2 import runner
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from decision.semantics import digest


def selected_fixture():
    result = {}
    for index, (alias, _) in enumerate(runner.SEED_ALIASES):
        for opponent in runner.OPPONENTS:
            for side in (0, 1):
                for policy in runner.ARMS:
                    key = (alias, opponent, side, policy)
                    name = '-'.join(map(str, key))
                    result[key] = {'row': {'game': name, 'seed_alias': alias, 'opponent': opponent,
                                           'side': side, 'policy': policy, 'seed': 11 + index},
                                   'replay': {'sha256': 'r' + name}, 'trace': {'sha256': 't' + name}}
    return result


PAIR_FIELDS = ('side', 'opponent', 'reference', 'first_branch_kind', 'first_branch_step',
    'branch_attribution', 'first_actor_request_components_differ', 'eligible_turns',
    'actor_observation_or_configuration_mismatch_turns', 'zero_both_selection_record_turns',
    'input_difference_turns', 'selection_record_count_difference_turns', 'equal_recorded_choices_turns',
    'choice_difference_turns', 'choice_difference_same_actor_request_turns',
    'choice_difference_different_actor_request_turns', 'choice_difference_same_request_same_task_multiset_turns',
    'matched_context_selection_records', 'teacher_label_consistent_selection_records',
    'teacher_label_disagreement_selection_records', 'candidate_teacher_agreement_selection_records',
    'reference_teacher_agreement_selection_records', 'first_choice_difference',
    'choice_difference_at_branch', 'post_branch_windows')


class FakeComparison:
    def __init__(self, value):
        self.value = value

    def finish(self):
        return deepcopy(self.value)


def loop_fixture(short=False, extra=False):
    selected = selected_fixture()
    first = next(iter(selected))
    for key, item in selected.items():
        label = '-'.join(map(str, key))
        item['replay']['path'] = 'replay/' + label
        item['trace'].update(path='trace/' + label, short=short and key == first, extra=extra and key == first)
        item['counts'] = {'task_forwards': 0}

    class Meter:
        def __init__(self, **_):
            self.opened, self.receipts = set(), []

        def snapshot(self):
            return {'files_opened': len(self.opened), 'files_completed': len(self.receipts)}

    def read_json(path, _metadata, meter):
        meter.opened.add(str(path))
        meter.receipts.append(str(path))
        return {}

    def trace_lines(path, metadata, meter):
        meter.opened.add(str(path))
        length = 718 if metadata['short'] else 720 if metadata['extra'] else 719
        for _ in range(length):
            yield {}
        meter.receipts.append(str(path))

    fake = types.SimpleNamespace(Meter=Meter, read_json=read_json, trace_lines=trace_lines,
        compact_replay=lambda _value, _meta: {}, descriptor=lambda: {},
        validate_trace=lambda _trace, _compact, _step, _descriptor: [])

    class Pair:
        def __init__(self, _candidate, _reference, side, opponent, name):
            self.side, self.opponent, self.name, self.turns = side, opponent, name, 0

        def observe_turn(self, step, _candidate, _reference):
            if step != self.turns:
                raise ValueError('Out-of-order synthetic trace')
            self.turns += 1

        def finish(self):
            if self.turns != 719:
                raise ValueError('Incomplete synthetic pair')
            value = dict.fromkeys(PAIR_FIELDS, 0)
            value.update(side=self.side, opponent=self.opponent, reference=self.name,
                         first_branch_kind='none', first_branch_step=None, branch_attribution='unknown',
                         first_actor_request_components_differ=None, first_choice_difference=None,
                         choice_difference_at_branch=None, post_branch_windows=None)
            return value

    return selected, fake, Pair


class RunnerTest(unittest.TestCase):
    def test_role_selection_rejects_missing_mismatched_and_overlapping_cases(self):
        selected = selected_fixture()
        self.assertEqual(len(selected), 12)
        runner.check_selected(selected)
        key = next(iter(selected))
        for mutation in ('missing', 'alias', 'side', 'policy', 'same_seed'):
            bad = deepcopy(selected)
            if mutation == 'missing':
                del bad[key]
            elif mutation == 'alias':
                bad[key]['row']['seed_alias'] = 'S03'
            elif mutation == 'side':
                bad[key]['row']['side'] = False
            elif mutation == 'policy':
                bad[key]['row']['policy'] = 'different'
            else:
                for candidate, item in bad.items():
                    if candidate[0] == 'S03':
                        item['row']['seed'] = 11
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.check_selected(bad)

    def test_private_snapshot_rejects_duplicate_game_identity(self):
        selected = selected_fixture()
        fake_safe = lambda _panel, *parts: '/'.join(parts)
        fake_read = lambda path: str(path).encode()
        with patch.object(runner, 'safe_file', side_effect=fake_safe), patch.object(runner, 'read_small', side_effect=fake_read):
            snapshot = runner.input_snapshot('panel', 'status', selected)
            self.assertEqual(len(snapshot['expected_inputs']), 12)
            keys = list(selected)
            selected[keys[1]]['row']['game'] = selected[keys[0]]['row']['game']
            with self.assertRaises(ValueError):
                runner.input_snapshot('panel', 'status', selected)

    def test_projection_rejects_unexpected_private_fields_and_renames_observation(self):
        value = dict.fromkeys(PAIR_FIELDS, 0)
        value.update(side=0, opponent='round2', reference='unweighted', first_branch_kind='actor',
                     first_branch_step=710, branch_attribution='unknown',
                     first_actor_request_components_differ=None, first_choice_difference=None,
                     choice_difference_at_branch=None)
        window = {'state_index': 711, 'semantic_physical_state_equal': True,
                  'own_bank_delta_changed': False, 'opponent_bank_delta_changed': False,
                  'margin_delta_changed': False}
        late = dict(window, state_index=719)
        value['post_branch_windows'] = {'1': dict(window), '24': late}
        result = runner.public_pair('S02', FakeComparison(value))
        self.assertTrue(result['post_branch_windows']['1']['observation_equal'])
        self.assertEqual(result['post_branch_windows']['1']['actions_elapsed'], 1)
        self.assertEqual(result['post_branch_windows']['24']['actions_elapsed'], 9)
        self.assertNotIn('semantic_physical_state_equal', result['post_branch_windows']['1'])
        extra = deepcopy(value)
        extra['private'] = 'synthetic'
        with self.assertRaises(ValueError):
            runner.public_pair('S02', FakeComparison(extra))
        extra = deepcopy(value)
        extra['first_choice_difference'] = {'step': 710, 'selection_record_index': 0,
            'candidate_op': 'PRIVATE_SENTINEL', 'reference_op': 'PLANT', 'same_unit': False,
            'same_target': True, 'candidate_request_kind': 'move', 'reference_request_kind': 'work',
            'selected_unit_request_equal': False, 'complete_actor_request_equal': False}
        with self.assertRaises(ValueError):
            runner.public_pair('S02', FakeComparison(extra))
        extra = deepcopy(value)
        extra['post_branch_windows']['1']['private'] = 'synthetic'
        with self.assertRaises(ValueError):
            runner.public_pair('S02', FakeComparison(extra))
        with self.assertRaises(ValueError):
            runner.public_pair('S04', FakeComparison(value))

    def test_invalid_cli_does_not_echo_private_argument(self):
        sentinel = 'PRIVATE_SENTINEL_ONLY_FOR_TEST'
        path = Path(__file__).resolve().parent / 'runner.py'
        result = subprocess.run([sys.executable, '-B', '-I', str(path), '--unknown', sentinel],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(sentinel, result.stdout + result.stderr)
        self.assertFalse(json.loads(result.stdout)['passed'])

    def test_shed_order_changes_pairing_even_when_v2_digest_is_equal(self):
        def frame(shed):
            return [{'observation': {'private': {'shed': shed, 'inventories': []}}},
                    {'observation': {'private': {'shed': {'x': 1}, 'inventories': []}}}]
        left = frame({'a': 1, 'b': 2})
        right = frame({'b': 2, 'a': 1})
        self.assertEqual(digest(left), digest(right))
        def wrap(item):
            base = digest(item)
            compact = {'full': [base] * 720, 'physical': [base] * 720}
            return runner.preserve_shed_order(compact, {'steps': [item] * 720})
        with patch.object(runner, 'analysis', types.SimpleNamespace(ordered_hash=ordered_hash), create=True):
            a, b = wrap(left), wrap(right)
        self.assertNotEqual(a['full'][0], b['full'][0])
        self.assertNotEqual(a['physical'][0], b['physical'][0])

    def test_four_triads_eight_pairs_and_each_arm_counted_once(self):
        selected, fake, pair = loop_fixture()
        progress = {'triads_completed': 0}
        with patch.object(runner, 'analysis', fake, create=True), patch.object(runner, 'PairComparison', pair, create=True), \
             patch.object(runner, 'checked_rows', return_value=selected), patch.object(runner, 'input_snapshot', return_value={}), \
             patch.object(runner, 'check_replay_seed'), patch.object(runner, 'check_trace_identity'), \
             patch.object(runner, 'preserve_shed_order', side_effect=lambda compact, _value: compact), redirect_stdout(io.StringIO()):
            result = runner.run(Path('synthetic-panel'), Path('synthetic-status'), {}, progress)
        self.assertEqual(progress['triads_completed'], 4)
        self.assertEqual((result['input_games'], result['paired_comparisons']), (12, 8))
        self.assertEqual(result['raw_reads'], {'files_opened': 24, 'files_completed': 24})
        self.assertEqual(len({(p['seed_alias'], p['side'], p['reference']) for p in result['pairs']}), 8)
        for counts in result['selection_counts'].values():
            self.assertEqual(counts['zero_selection_turns'], 4 * 719)

    def test_short_or_extra_trace_fails_before_first_triad_completes(self):
        for mode in ('short', 'extra'):
            with self.subTest(mode=mode):
                selected, fake, pair = loop_fixture(short=mode == 'short', extra=mode == 'extra')
                progress = {'triads_completed': 0}
                with patch.object(runner, 'analysis', fake, create=True), patch.object(runner, 'PairComparison', pair, create=True), \
                     patch.object(runner, 'checked_rows', return_value=selected), patch.object(runner, 'input_snapshot', return_value={}), \
                     patch.object(runner, 'check_replay_seed'), patch.object(runner, 'check_trace_identity'), \
                     patch.object(runner, 'preserve_shed_order', side_effect=lambda compact, _value: compact), redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):
                        runner.run(Path('synthetic-panel'), Path('synthetic-status'), {}, progress)
                self.assertEqual(progress['triads_completed'], 0)


if __name__ == '__main__':
    unittest.main()
