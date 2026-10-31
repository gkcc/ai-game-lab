"""Synthetic orchestration regressions; no private data, model or job."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from learning.task_pair_fresh_multi_v1 import entry
from learning.task_student_anchor_collection_v1 import sources
from learning.task_teacher_cross_source_v1 import archive
from learning.task_teacher_dev_deviation_v1 import membership
from learning.task_water_residual_v1.test_policy import metadata_and_inputs
from learning.task_multisource_rank_v1 import core, dedup, evaluate, experiment


def screen_inputs(*, count=20, old_drop=1, net=2, role_drop=False,
                  missing_role=False, changed=1):
    new = dict(overall=dict(groups=20, baseline_correct=10, residual_correct=10 + net, changed=changed),
        by_role={role: dict(overall=dict(groups=10, baseline_correct=5, residual_correct=6))
                 for role in ('A07', 'A08')})
    if role_drop:
        new['by_role']['A07']['overall'] = dict(groups=10, baseline_correct=6, residual_correct=5)
        new['by_role']['A08']['overall'] = dict(groups=10, baseline_correct=4, residual_correct=5 + net)
    if missing_role:
        new['by_role']['A07']['overall'] = dict(groups=20, baseline_correct=10, residual_correct=10 + net)
        new['by_role']['A08']['overall'] = dict(groups=0, baseline_correct=0, residual_correct=0)
    old_base = count // 2
    old = dict(by_target_operation={op: dict(groups=0, baseline_correct=0, residual_correct=0)
                                   for op in ('DIG', 'FERTILIZE', 'HARVEST', 'PLANT', 'WATER')})
    old['by_target_operation']['DIG'] = dict(groups=count, baseline_correct=old_base,
                                              residual_correct=old_base - old_drop)
    return new, old


class SelectionTests(unittest.TestCase):
    def test_fixed_selection_uses_only_source_and_population(self):
        # The index selector has no group input and should perform no file I/O.
        with patch('builtins.open', side_effect=AssertionError('unexpected file access')):
            first = experiment._indices('old', 500)
            again = experiment._indices('old', 500)
            second = experiment._indices('new', 500)
        self.assertEqual(first, again)
        for train, diagnostic in (first, second):
            self.assertEqual((len(train), len(diagnostic)), (256, 128))
            self.assertTrue(train.isdisjoint(diagnostic))
            self.assertTrue(train | diagnostic <= set(range(500)))
        self.assertNotEqual(first, second)

    def test_old_source_selection_returns_only_after_both_archive_seals(self):
        calls = []
        candidate = np.zeros((1, 98), dtype=np.float32)
        binding = dict(earlier=dict(historical=dict(archive_sha256={'old': 'first', 'new': 'second'})))
        def scan(_path, *, expected_sha256, expected_groups, expected_candidates, guard, consume):
            calls.append(expected_sha256)
            for index in range(expected_groups):
                guard()
                # The float feature is a synthetic index only, never real data.
                consume(index, np.full(275, index, dtype=np.float32), candidate, 0)
            return dict(groups=expected_groups, candidates=expected_candidates, verified_after_scan=True)
        with patch.object(archive, 'scan_archive', side_effect=scan):
            train, diagnostic, roles = experiment._collect_old(binding, lambda: None)
        self.assertEqual(calls, ['first', 'second'])
        self.assertEqual((len(train), len(diagnostic), len(roles)), (512, 256, 256))
        for name, train_slice, diag_slice in (('old', train[:256], diagnostic[:128]),
                                              ('new', train[256:], diagnostic[128:])):
            selected, measured = experiment._indices(name, experiment.cross.GROUPS[name])
            self.assertEqual({int(item.shared[0]) for item in train_slice}, selected)
            self.assertEqual({int(item.shared[0]) for item in diag_slice}, measured)
        self.assertEqual(roles, ['T01'] * 128 + ['T02'] * 128)

    def test_failed_archive_seal_stops_without_returning_selected_groups(self):
        binding = dict(earlier=dict(historical=dict(archive_sha256={'old': 'first', 'new': 'second'})))
        for failed_position in (1, 2):
            calls = []
            def scan(_path, *, expected_sha256, expected_groups, expected_candidates, guard, consume):
                calls.append(expected_sha256)
                for index in range(expected_groups):
                    consume(index, np.zeros(275, dtype=np.float32), np.zeros((1, 98), dtype=np.float32), 0)
                if len(calls) == failed_position:
                    raise ValueError('synthetic final rehash failure')
                return dict(groups=expected_groups, candidates=expected_candidates, verified_after_scan=True)
            with self.subTest(failed_position=failed_position), patch.object(archive, 'scan_archive', side_effect=scan):
                with self.assertRaisesRegex(ValueError, 'synthetic final rehash failure'):
                    experiment._collect_old(binding, lambda: None)
                self.assertEqual(calls, ['first', 'second'][:failed_position])

    def test_unverified_receipt_or_missing_selected_index_fails_closed(self):
        binding = dict(earlier=dict(historical=dict(archive_sha256={'old': 'first', 'new': 'second'})))
        for fault in ('receipt', 'missing'):
            calls = []
            omitted = min(experiment._indices('new', experiment.cross.GROUPS['new'])[0])
            def scan(_path, *, expected_sha256, expected_groups, expected_candidates, guard, consume):
                calls.append(expected_sha256)
                for index in range(expected_groups):
                    if fault == 'missing' and expected_sha256 == 'second' and index == omitted:
                        continue
                    consume(index, np.zeros(275, np.float32), np.zeros((1, 98), np.float32), 0)
                return dict(groups=expected_groups, candidates=expected_candidates,
                            verified_after_scan=not (fault == 'receipt' and expected_sha256 == 'second'))
            with self.subTest(fault=fault), patch.object(archive, 'scan_archive', side_effect=scan):
                with self.assertRaises(ValueError): experiment._collect_old(binding, lambda: None)
                self.assertEqual(calls, ['first', 'second'])


class ScreenTests(unittest.TestCase):
    def test_thresholds_are_inclusive(self):
        new, old = screen_inputs(count=20, old_drop=1, net=2)
        result = experiment._screen(new, old)
        self.assertEqual(result['screen'], 'passed')
        self.assertTrue(all(result['checks'].values()))
        for kwargs in (dict(count=20, old_drop=2), dict(net=1), dict(changed=0)):
            with self.subTest(condition=kwargs):
                self.assertEqual(experiment._screen(*screen_inputs(**kwargs))['screen'], 'failed')

    def test_insufficient_coverage_is_unknown_but_observed_drop_is_failed(self):
        result = experiment._screen(*screen_inputs(count=19, old_drop=1))
        self.assertEqual(result['screen'], 'unknown')
        self.assertFalse(result['checks']['old_non_water_coverage_at_least_twenty'])
        self.assertIsNone(result['checks']['old_non_water_drop_at_most_five_points'])
        observed_drop = experiment._screen(*screen_inputs(count=19, role_drop=True))
        self.assertEqual(observed_drop['screen'], 'failed')
        self.assertFalse(observed_drop['checks']['each_new_dev_role_no_accuracy_drop'])

    def test_missing_development_role_unknown_unless_other_failure_observed(self):
        incomplete = experiment._screen(*screen_inputs(missing_role=True))
        self.assertEqual(incomplete['screen'], 'unknown')
        self.assertIsNone(incomplete['checks']['each_new_dev_role_no_accuracy_drop'])
        failed = experiment._screen(*screen_inputs(missing_role=True, old_drop=2))
        self.assertEqual(failed['screen'], 'failed')


class SummaryTests(unittest.TestCase):
    @staticmethod
    def genuine_summary(source, roles, *, count=2, group_roles=False, assigned_roles=None):
        metadata, _ = metadata_and_inputs()
        shared = np.zeros(275, dtype=np.float32)
        candidates = np.zeros((2, 98), dtype=np.float32)
        for index, op in enumerate(('DIG', 'WATER')):
            candidates[index, metadata['feature_names'].index('task.op.' + op)] = 1
            candidates[index, metadata['feature_names'].index('unit.index')] = index
        groups = [core.Group(shared, candidates, np.array([2., 1.], np.float64), index % 2, source)
                  for index in range(count)]
        selected_roles = (assigned_roles if assigned_roles is not None
                          else [roles[(index // 2 if group_roles else index) % 2] for index in range(count)])
        weights = np.zeros(373, dtype=np.float32)
        weights[275 + metadata['feature_names'].index('unit.index')] = 2
        return evaluate.assess(groups, np.zeros(373, np.float64), np.ones(373, np.float64),
                               weights, metadata, roles=selected_roles, role_labels=roles)

    def test_real_evaluate_aggregates_pass_and_wrong_counts_or_nll_fail(self):
        roles = ('T01', 'T02')
        actual = self.genuine_summary('old', roles)
        self.assertEqual(experiment._validate_summary(actual, roles, 'old'), 2)
        with self.assertRaises(ValueError): experiment._validate_summary(actual, roles, 'new')
        for mutation in ('count', 'nll', 'nan', 'empty_loss'):
            changed = deepcopy(actual)
            if mutation == 'count':
                changed['by_target_operation']['DIG']['groups'] += 1
            elif mutation == 'nll':
                changed['overall']['baseline_nll'] += 1
            elif mutation == 'nan':
                changed['overall']['residual_nll'] = float('nan')
            else:
                changed['by_target_operation']['PLANT']['baseline_nll'] = 0.0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                experiment._validate_summary(changed, roles, 'old')

    def test_role_operation_cross_partition_tampering_is_rejected(self):
        roles = ('T01', 'T02')
        actual = self.genuine_summary('old', roles)
        self.assertEqual(experiment._validate_summary(actual, roles, 'old'), 2)
        changed = deepcopy(actual)
        by_op = changed['by_role']['T01']['by_target_operation']
        # Keep the role's total and weighted NLL intact while moving an entire
        # observed bucket to a different operation; global marginals remain
        # untouched, so only the role-by-operation cross-check catches this.
        by_op['DIG'], by_op['WATER'] = by_op['WATER'], by_op['DIG']
        with self.assertRaises(ValueError): experiment._validate_summary(changed, roles, 'old')

    def test_role_operation_nll_tampering_is_rejected_even_if_role_mean_balances(self):
        roles = ('T01', 'T02')
        actual = self.genuine_summary('old', roles, count=4, group_roles=True)
        self.assertEqual(experiment._validate_summary(actual, roles, 'old'), 4)
        changed = deepcopy(actual)
        by_op = changed['by_role']['T01']['by_target_operation']
        by_op['DIG']['baseline_nll'] += 0.2
        by_op['WATER']['baseline_nll'] -= 0.2
        with self.assertRaises(ValueError): experiment._validate_summary(changed, roles, 'old')

    def test_result_rejects_dedup_and_single_candidate_accounting_tamper(self):
        old = self.genuine_summary('old', ('T01', 'T02'), count=32)
        new = self.genuine_summary('new', ('A07', 'A08'), count=32)
        pools = ('old_train', 'new_train', 'old_diag', 'new_dev')
        incoming = dict(zip(pools, (512, 59, 256, 62)))
        retained = dict(zip(pools, (512, 59, 32, 32)))
        removed = {key: incoming[key] - retained[key] for key in pools}
        zero = {key: 0 for key in pools}
        report = dict(input_count=889, retained_count=sum(retained.values()), input_by_pool=incoming,
            retained_by_pool=retained, duplicate_removed_by_pool=removed,
            conflict_removed_by_pool=zero.copy(), single_candidate_excluded_by_pool=zero.copy(),
            constant_normalized_excluded_by_pool=zero.copy(), eligible_by_pool=retained.copy(),
            eligible_by_role={'old_train': {'T01': 256, 'T02': 256}, 'new_train': {'A01': 30, 'A02': 29},
                              'old_diag': {'T01': 16, 'T02': 16}, 'new_dev': {'A07': 16, 'A08': 16}})
        head = 'a' * 40
        result = dict(schema='task-multisource-rank-result-v1', registration_commit=head,
            new_dev=new, old_diag=old, dedup=report,
            parity=dict(groups=64, allclose=True, adjusted_argmax_mismatches=0, zero_residual_mismatches=0),
            screen=experiment._screen(new, old), weights_sha256='b' * 64, baseline_sha256='c' * 64,
            updates_completed=64, nonzero_gradient_seen=True, weights_changed=True, device='cpu',
            new_games=0, independent_evaluation=False, software={})
        encode = lambda value: json.dumps(value, allow_nan=False).encode()
        self.assertEqual(experiment._result(encode(result), head), result)
        for field in ('input_by_pool', 'single_candidate_excluded_by_pool',
                      'constant_normalized_excluded_by_pool', 'eligible_by_pool'):
            changed = deepcopy(result)
            changed['dedup'][field]['old_diag'] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                experiment._result(encode(changed), head)
        changed = deepcopy(result)
        # Preserve the global input and retained totals, changing only which
        # training pool they are attributed to; exact per-pool sizes still gate.
        for field in ('input_by_pool', 'retained_by_pool', 'eligible_by_pool'):
            changed['dedup'][field]['old_train'] -= 1
            changed['dedup'][field]['new_train'] += 1
        changed['dedup']['eligible_by_role']['old_train']['T01'] -= 1
        changed['dedup']['eligible_by_role']['new_train']['A01'] += 1
        with self.assertRaises(ValueError): experiment._result(encode(changed), head)
        changed = deepcopy(result)
        changed['dedup']['single_candidate_excluded_by_pool']['old_train'] = 497
        changed['dedup']['eligible_by_pool']['old_train'] = 15
        changed['dedup']['eligible_by_role']['old_train'] = {'T01': 7, 'T02': 8}
        with self.assertRaises(ValueError): experiment._result(encode(changed), head)
        changed = deepcopy(result)
        changed['new_dev'] = self.genuine_summary('new', ('A07', 'A08'), count=32,
                                                  assigned_roles=['A07'] * 25 + ['A08'] * 7)
        changed['dedup']['eligible_by_role']['new_dev'] = {'A07': 25, 'A08': 7}
        changed['screen'] = experiment._screen(changed['new_dev'], changed['old_diag'])
        with self.assertRaises(ValueError): experiment._result(encode(changed), head)
        changed = deepcopy(result)
        changed['parity']['groups'] -= 1
        with self.assertRaises(ValueError): experiment._result(encode(changed), head)
        for pool, first, second in (('old_diag', 'T01', 'T02'), ('new_dev', 'A07', 'A08')):
            changed = deepcopy(result)
            changed['dedup']['eligible_by_role'][pool][first] -= 1
            changed['dedup']['eligible_by_role'][pool][second] += 1
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                experiment._result(encode(changed), head)
        changed = deepcopy(result)
        changed['dedup']['eligible_by_role']['old_train']['T01'] -= 1
        with self.assertRaises(ValueError): experiment._result(encode(changed), head)


class SnapshotTests(unittest.TestCase):
    def test_caller_arrays_are_immutable_snapshots_before_deduplication(self):
        old_shared = np.zeros(275, np.float32)
        old_candidates = np.zeros((2, 98), np.float32)
        old_candidates[1, 0] = 1
        old_item = SimpleNamespace(shared=old_shared, candidates=old_candidates, target_index=0)
        old_train, old_diag = [old_item] * 512, [old_item] * 256
        train_features, dev_features = np.zeros((2, 373), np.float32), np.zeros((2, 373), np.float32)
        train_features[1, 275], dev_features[1, 275] = 1, 1
        new_train = [SimpleNamespace(features=train_features, target_index=0, role='A01')] * 59
        new_dev = ([SimpleNamespace(features=dev_features, target_index=0, role='A07')] * 31
                   + [SimpleNamespace(features=dev_features, target_index=0, role='A08')] * 31)
        def fake_deduplicate(items, *, mean, scale, guard):
            old_shared[0], old_candidates[0, 0] = 9, 9
            train_features[0, 0], train_features[0, 275] = 9, 9
            dev_features[0, 0], dev_features[0, 275] = 9, 9
            for index in (0, 512, 512 + 59, 512 + 59 + 256):
                item = items[index]
                self.assertEqual(float(item.shared[0]), 0.)
                self.assertEqual(float(item.candidates[0, 0]), 0.)
                self.assertFalse(item.shared.flags.writeable)
                self.assertFalse(item.candidates.flags.writeable)
                with self.assertRaises(ValueError): item.shared.setflags(write=True)
                with self.assertRaises(ValueError): item.candidates.setflags(write=True)
            return items, {}
        with patch.object(dedup, 'deduplicate', side_effect=fake_deduplicate):
            pools, summary = experiment._deduplicate(old_train, old_diag, ['T01'] * 128 + ['T02'] * 128,
                new_train, new_dev, np.zeros(373, np.float64), np.ones(373, np.float64), lambda: None)
        self.assertEqual(summary['eligible_by_pool'],
                         {'old_train': 512, 'new_train': 59, 'old_diag': 256, 'new_dev': 62})
        self.assertEqual(float(pools['new_train'][0].shared[0]), 0.)

    def test_single_and_constant_exclusion_is_balanced_without_replacement(self):
        numbers = iter(range(1, 890))
        metadata, _ = metadata_and_inputs()
        unit = metadata['feature_names'].index('unit.index')
        target_x = metadata['feature_names'].index('target.x')
        dig = metadata['feature_names'].index('task.op.DIG')
        def old(single=False, constant=False):
            marker = next(numbers)
            candidates = np.zeros((1 if single else 2, 98), np.float32)
            candidates[:, dig] = 1
            if not single: candidates[1, unit if constant else target_x] = 1
            return SimpleNamespace(shared=np.full(275, marker, np.float32),
                                   candidates=candidates, target_index=0)
        def new(role, single=False, constant=False):
            marker = next(numbers)
            features = np.zeros((1 if single else 2, 373), np.float32)
            features[:, :275] = marker
            features[:, 275 + dig] = 1
            if not single: features[1, 275 + (unit if constant else target_x)] = 1
            return SimpleNamespace(features=features, target_index=0, role=role)
        old_train = [old(single=(index == 0), constant=(index == 1)) for index in range(512)]
        old_diag = [old(single=(index == 0), constant=(index == 1)) for index in range(256)]
        new_train = [new('A01', single=(index == 0), constant=(index == 1)) for index in range(59)]
        new_dev = [new('A07' if index < 31 else 'A08', single=(index == 0), constant=(index == 1))
                   for index in range(62)]
        excluded = {float(group.shared[0]) for group in old_train[:2] + old_diag[:2]}
        excluded.update(float(group.features[0, 0]) for group in new_train[:2] + new_dev[:2])
        mean, scale = np.zeros(373, np.float64), np.ones(373, np.float64)
        mean[275 + unit] = 2**30
        pools, report = experiment._deduplicate(old_train, old_diag, ['T01'] * 128 + ['T02'] * 128,
            new_train, new_dev, mean, scale, lambda: None)
        expected = {'old_train': 510, 'new_train': 57, 'old_diag': 254, 'new_dev': 60}
        self.assertEqual(report['input_count'], 889)
        self.assertEqual(report['retained_count'], 889)
        self.assertEqual(report['eligible_by_pool'], expected)
        self.assertEqual(report['single_candidate_excluded_by_pool'], {key: 1 for key in expected})
        self.assertEqual(report['constant_normalized_excluded_by_pool'], {key: 1 for key in expected})
        self.assertEqual(sum(report['duplicate_removed_by_pool'].values()), 0)
        self.assertEqual(sum(report['conflict_removed_by_pool'].values()), 0)
        self.assertEqual({key: len(value) for key, value in pools.items()}, expected)
        self.assertTrue(excluded.isdisjoint(float(item.shared[0]) for group in pools.values() for item in group))
        groups = [core.Group(item.shared, item.candidates, np.array([1., 0.], np.float64),
                             item.target, 'old') for item in pools['old_diag']]
        diagnostic = evaluate.assess(groups, mean, scale, np.zeros(373, np.float32), metadata,
                                    roles=[item.role for item in pools['old_diag']],
                                    role_labels=('T01', 'T02'))
        new_summary, _ = screen_inputs()
        self.assertEqual(experiment._screen(new_summary, diagnostic)['old_non_water_groups'], expected['old_diag'])

    def test_inadequate_coverage_is_reported_anonymously_before_stopping(self):
        old_candidates = np.zeros((2, 98), np.float32)
        old_candidates[1, 0] = 1
        old = SimpleNamespace(shared=np.zeros(275, np.float32), candidates=old_candidates, target_index=0)
        old_train, old_diag = [old] * 512, [old] * 256
        single, multi = np.zeros((1, 373), np.float32), np.zeros((2, 373), np.float32)
        multi[1, 275] = 1
        new_train = [SimpleNamespace(features=single, target_index=0, role='A01')] * 59
        new_dev = ([SimpleNamespace(features=multi, target_index=0, role='A07')] * 31
                   + [SimpleNamespace(features=multi, target_index=0, role='A08')] * 31)
        counts = {'old_train': 512, 'new_train': 59, 'old_diag': 256, 'new_dev': 62}
        def fake_dedup(items, *, mean, scale, guard):
            return items, dict(input_count=889, retained_count=889, input_by_pool=counts.copy(),
                retained_by_pool=counts.copy(), duplicate_removed_by_pool={key: 0 for key in counts},
                conflict_removed_by_pool={key: 0 for key in counts}, component_count=889, conflict_components=0)
        reports = []
        with patch.object(dedup, 'deduplicate', side_effect=fake_dedup):
            with self.assertRaisesRegex(ValueError, 'coverage insufficient'):
                experiment._deduplicate(old_train, old_diag, ['T01'] * 128 + ['T02'] * 128,
                    new_train, new_dev, np.zeros(373, np.float64), np.ones(373, np.float64), lambda: None,
                    on_report=lambda report: reports.append(deepcopy(report)))
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]['eligible_by_pool']['new_train'], 0)
        self.assertEqual(reports[0]['single_candidate_excluded_by_pool']['new_train'], 59)
        self.assertEqual(reports[0]['eligible_by_role']['new_train'], {'A01': 0, 'A02': 0})
        self.assertEqual(reports[0]['eligible_by_role']['old_diag'], {'T01': 128, 'T02': 128})
        serialized = json.dumps(reports[0])
        for forbidden in ('features', 'target', 'seed', 'logits', 'PRIVATE_MARKER'):
            self.assertNotIn(forbidden, serialized)


class CastReportTests(unittest.TestCase):
    def test_finite_stable_and_changed_first_choices_are_counted(self):
        groups = [SimpleNamespace(source='old', baseline=np.array([1., 2.], np.float64)),
                  SimpleNamespace(source='new', baseline=np.array([1., 1. + 1e-8], np.float64)),
                  SimpleNamespace(source='new', baseline=np.array([1e300, 0.], np.float64))]
        result = experiment._baseline_cast_report(groups, lambda: None)
        self.assertEqual(result, {'old': {'groups': 1, 'nonfinite': 0, 'argmax_changed': 0},
                                  'new': {'groups': 2, 'nonfinite': 1, 'argmax_changed': 1}})
        self.assertNotIn('baseline', json.dumps(result))
        with self.assertRaises(ValueError):
            experiment._validate_cast_report(result, {'eligible_by_pool': {'old_train': 1, 'new_train': 2}})

    def test_successful_cast_receipt_requires_both_sources_and_exact_counts(self):
        groups = [SimpleNamespace(source='old', baseline=np.array([2., 1.], np.float64)),
                  SimpleNamespace(source='new', baseline=np.array([2., 1.], np.float64))]
        report = experiment._baseline_cast_report(groups, lambda: None)
        coverage = {'eligible_by_pool': {'old_train': 1, 'new_train': 1, 'old_diag': 1, 'new_dev': 1},
                    'eligible_by_role': {'old_train': {'T01': 1, 'T02': 0}, 'new_train': {'A01': 1, 'A02': 0},
                                         'old_diag': {'T01': 1, 'T02': 0}, 'new_dev': {'A07': 1, 'A08': 0}}}
        self.assertIsNone(experiment._validate_cast_report(report, coverage))
        variants = []
        missing_source = deepcopy(report); del missing_source['new']; variants.append(missing_source)
        missing_field = deepcopy(report); del missing_field['old']['nonfinite']; variants.append(missing_field)
        changed = deepcopy(report); changed['old']['argmax_changed'] = 1; variants.append(changed)
        nonfinite = deepcopy(report); nonfinite['new']['nonfinite'] = 1; variants.append(nonfinite)
        wrong_count = deepcopy(report); wrong_count['old']['groups'] = 2; variants.append(wrong_count)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                experiment._validate_cast_report(value, coverage)


class ClosedReceiptTests(unittest.TestCase):
    @staticmethod
    def fixture():
        head, binding_sha = 'a' * 40, 'b' * 64
        coverage = {'eligible_by_pool': {'old_train': 1, 'new_train': 1, 'old_diag': 1, 'new_dev': 1},
                    'eligible_by_role': {'old_train': {'T01': 1, 'T02': 0}, 'new_train': {'A01': 1, 'A02': 0},
                                         'old_diag': {'T01': 1, 'T02': 0}, 'new_dev': {'A07': 1, 'A08': 0}}}
        cast = {'old': {'groups': 1, 'nonfinite': 0, 'argmax_changed': 0},
                'new': {'groups': 1, 'nonfinite': 0, 'argmax_changed': 0}}
        attempt = dict(schema='task-multisource-rank-attempt-v1', commit=head, retry_allowed=False)
        claim = dict(schema='task-multisource-rank-claim-v1', commit=head, attempt=1,
                     freeze_sha256=binding_sha, job_token_sha256='c' * 64, retry_allowed=False)
        child = dict(schema='task-multisource-rank-child-attempt-v1', commit=head, retry_allowed=False)
        owner = dict(schema='task-pair-fresh-multi-owner-v1', registration_commit=head,
                     binding_sha256=binding_sha, pid=1, created_us=1, deadline=1.0)
        encode = lambda value: json.dumps(value, sort_keys=True).encode()
        files = {'worker-result.local.json': b'{"synthetic":true}', 'job.local.json': b'{"ok":true}',
                 'attempt.local.json': encode(attempt), 'owner.local.json': encode(owner),
                 'claim.local.json': encode(claim), 'child-attempt.local.json': encode(child),
                 'coverage.local.json': encode(coverage), 'baseline-cast.local.json': encode(cast),
                 'residual.local.npy': b'synthetic-residual', 'baseline.local.npz': b'synthetic-baseline'}
        links = {'worker-result.local.json': 'worker_sha256', 'job.local.json': 'job_sha256',
                 'attempt.local.json': 'attempt_sha256', 'owner.local.json': 'owner_sha256',
                 'claim.local.json': 'claim_sha256', 'child-attempt.local.json': 'child_attempt_sha256',
                 'coverage.local.json': 'coverage_sha256', 'baseline-cast.local.json': 'cast_sha256',
                 'residual.local.npy': 'weights_sha256', 'baseline.local.npz': 'baseline_sha256'}
        closed = dict(schema='task-multisource-rank-closed-v1', registration_commit=head,
                      retry_allowed=False, binding_sha256=binding_sha)
        closed.update({field: experiment._sha(files[name]) for name, field in links.items()})
        result = dict(dedup=deepcopy(coverage), weights_sha256=closed['weights_sha256'],
                      baseline_sha256=closed['baseline_sha256'])
        binding = {'earlier': {'baseline_sha256': closed['baseline_sha256']}}
        files['closed.local.json'] = encode(closed)
        return SimpleNamespace(head=head, binding=binding, binding_sha=binding_sha, files=files,
                               result=result, closed=closed, cast=cast, coverage=coverage, encode=encode)

    def load(self, case):
        def read(path, limit=None):
            name = Path(path).name
            if name not in case.files: raise FileNotFoundError('synthetic missing receipt')
            return case.files[name]
        with ExitStack() as stack:
            for operation in (
                patch.object(experiment, 'PRIVATE', Path('synthetic-stage')),
                patch.object(experiment, '_read', side_effect=read),
                patch.object(experiment, '_registered_binding', return_value=(case.binding, case.binding_sha)),
                patch.object(experiment, '_result', return_value=case.result),
                patch.object(experiment.transfer, '_job_ok', return_value=True),
                patch.object(Path, 'exists', lambda path: Path(path).name in case.files),
            ):
                stack.enter_context(operation)
            return experiment.load_completed()

    def test_closed_receipts_accept_matching_cast_and_coverage(self):
        case = self.fixture()
        self.assertEqual(self.load(case), case.result)

    def test_missing_or_stale_cast_and_coverage_cannot_be_consumed(self):
        for missing in ('baseline-cast.local.json', 'coverage.local.json'):
            case = self.fixture(); del case.files[missing]
            with self.subTest(missing=missing), self.assertRaises(FileNotFoundError): self.load(case)
        stale = self.fixture()
        stale.files['baseline-cast.local.json'] += b' '
        with self.assertRaises(ValueError): self.load(stale)
        for field in ('groups', 'argmax_changed', 'nonfinite'):
            case = self.fixture()
            case.cast['old'][field] += 1
            case.files['baseline-cast.local.json'] = case.encode(case.cast)
            # Even a self-consistent receipt checksum must not authorize a
            # count mismatch or a detected cast hazard.
            case.closed['cast_sha256'] = experiment._sha(case.files['baseline-cast.local.json'])
            case.files['closed.local.json'] = case.encode(case.closed)
            with self.subTest(field=field), self.assertRaises(ValueError): self.load(case)
        case = self.fixture()
        case.coverage['eligible_by_pool']['old_train'] = 2
        case.files['coverage.local.json'] = case.encode(case.coverage)
        case.closed['coverage_sha256'] = experiment._sha(case.files['coverage.local.json'])
        case.files['closed.local.json'] = case.encode(case.closed)
        with self.assertRaises(ValueError): self.load(case)


class ChildAuthorizationTests(unittest.TestCase):
    def exercise(self, *, bad_claim=False, reject_job=False, reject_owner=False,
                 fail_collect=False, fail_coverage=False):
        head, binding_sha = 'a' * 40, 'b' * 64
        binding = dict(earlier=dict(train_sha256='train', dev_sha256='dev', baseline_sha256='base',
                                    schema_sha256='semantic', historical=dict(schema_sha256='history')))
        claim = dict(schema='task-multisource-rank-claim-v1', commit=head,
                     attempt=2 if bad_claim else 1, freeze_sha256=binding_sha, retry_allowed=False)
        files = {'claim.local.json': claim,
                 'attempt.local.json': dict(schema='task-multisource-rank-attempt-v1',
                                            commit=head, retry_allowed=False),
                 'owner.local.json': {'synthetic': True}}
        class StoppedAtData(Exception): pass
        class StoppedAtScan(Exception): pass
        job = Mock(side_effect=ValueError('synthetic job denial') if reject_job else None)
        owner = Mock(side_effect=ValueError('synthetic owner denial') if reject_owner else None)
        model = Mock(side_effect=AssertionError('real model must not be reached'))
        fake_train, fake_dev = [None] * 59, [None] * 62
        contract = dict(registration_commit=experiment.cross.DATA_COMMIT,
                        train_sha256='train', dev_sha256='dev')
        data = Mock(side_effect=None if (fail_collect or fail_coverage) else StoppedAtData(),
                    return_value=(fake_train, fake_dev, contract))
        collect = Mock(side_effect=None if fail_coverage else StoppedAtScan(), return_value=([], [], []))
        def no_coverage(*_args, on_report):
            on_report(dict(eligible_by_pool={'old_train': 0},
                           eligible_by_role={'old_train': {'T01': 0, 'T02': 0}}))
            raise ValueError('synthetic coverage gate')
        deduplicate = Mock(side_effect=no_coverage if fail_coverage else AssertionError('dedup must not be reached'))
        context = SimpleNamespace(task_sha256='base', model_normalization=lambda: ('mean', 'scale', 'meta'))

        def read(path, _limit=None):
            return json.dumps(files[Path(path).name]).encode()
        def write(path, value):
            name = Path(path).name
            if name in files: raise FileExistsError('synthetic exclusive marker')
            files[name] = value

        with ExitStack() as stack:
            for operation in (
                patch.object(experiment, 'PRIVATE', Path('synthetic-stage')),
                patch.object(experiment, '_runtime_binding', return_value=(head, binding, binding_sha)),
                patch.object(experiment, '_read', side_effect=read),
                patch.object(experiment.previous, '_write_new', side_effect=write),
                patch.object(experiment.previous, '_guard'),
                patch.object(experiment.transfer, 'load_trainable', data),
                patch.object(sources, 'open_context', return_value=context),
                patch.object(experiment.cross, '_schema_identity', return_value='semantic'),
                patch.object(experiment, '_collect_old', collect),
                patch.object(experiment, '_deduplicate', deduplicate),
                patch.object(experiment, '_model', model),
                patch.object(membership, 'verify_worker', job),
                patch.object(entry, 'verify_live_owner', owner),
                patch.object(Path, 'exists', lambda path: Path(path).name in files),
                patch.dict(os.environ, {membership.ENV_KEY: 'synthetic-job'}),
            ):
                stack.enter_context(operation)
            try:
                experiment.child()
                error = None
            except Exception as exc:
                error = exc
        return SimpleNamespace(files=files, error=error, job=job, owner=owner, data=data,
                               collect=collect, deduplicate=deduplicate, model=model,
                               head=head, binding_sha=binding_sha, claim=claim)

    def test_bad_claim_denies_before_job_owner_or_data(self):
        result = self.exercise(bad_claim=True)
        self.assertIsInstance(result.error, ValueError)
        result.job.assert_not_called(); result.owner.assert_not_called(); result.data.assert_not_called()
        self.assertNotIn('child-attempt.local.json', result.files)
        result.model.assert_not_called()

    def test_job_and_owner_denial_prevent_any_source_access(self):
        rejected_job = self.exercise(reject_job=True)
        self.assertIsInstance(rejected_job.error, ValueError)
        rejected_job.job.assert_called_once()
        rejected_job.owner.assert_not_called(); rejected_job.data.assert_not_called()
        self.assertNotIn('child-attempt.local.json', rejected_job.files)
        rejected_owner = self.exercise(reject_owner=True)
        self.assertIsInstance(rejected_owner.error, ValueError)
        rejected_owner.job.assert_called_once(); rejected_owner.owner.assert_called_once()
        rejected_owner.data.assert_not_called(); rejected_owner.model.assert_not_called()
        self.assertNotIn('child-attempt.local.json', rejected_owner.files)

    def test_valid_authorization_claimed_before_data_and_failure_is_sealed(self):
        result = self.exercise()
        self.assertEqual(type(result.error).__name__, 'StoppedAtData')
        result.job.assert_called_once(); result.owner.assert_called_once(); result.data.assert_called_once()
        result.job.assert_called_once_with('synthetic-job', result.claim, result.binding_sha)
        self.assertEqual(result.owner.call_args.kwargs['commit'], result.head)
        self.assertEqual(result.owner.call_args.kwargs['binding_sha'], result.binding_sha)
        self.assertIn('child-attempt.local.json', result.files)
        self.assertIn('worker-failed.local.json', result.files)
        self.assertNotIn('worker-result.local.json', result.files)
        result.model.assert_not_called()

    def test_source_failure_never_reaches_model(self):
        result = self.exercise(fail_collect=True)
        self.assertEqual(type(result.error).__name__, 'StoppedAtScan')
        result.collect.assert_called_once(); result.model.assert_not_called()
        self.assertIn('worker-failed.local.json', result.files)
        self.assertNotIn('worker-result.local.json', result.files)

    def test_child_preserves_anonymous_coverage_before_coverage_gate_failure(self):
        result = self.exercise(fail_coverage=True)
        self.assertIsInstance(result.error, ValueError)
        self.assertEqual(result.files['coverage.local.json']['eligible_by_pool']['old_train'], 0)
        self.assertEqual(result.files['worker-failed.local.json']['phase'], 'dedup')
        self.assertNotIn('worker-result.local.json', result.files)
        result.deduplicate.assert_called_once(); result.model.assert_not_called()
        for forbidden in ('features', 'target', 'seed', 'logits'):
            self.assertNotIn(forbidden, json.dumps(result.files['coverage.local.json']))


if __name__ == '__main__':
    unittest.main()
