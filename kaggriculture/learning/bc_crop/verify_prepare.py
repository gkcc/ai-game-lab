"""Preparation/interpretation regressions; no games or training are launched.

Supply an already-owned scratch directory. All test outputs remain inside it.
The caller must clean that directory after this process exits.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from .agent import ROOT, sha
from .experiment import integrity, read, save
from .prepare import SEED_KEYS, novelty, seed_config
from .report import interpretation


class Preparation(unittest.TestCase):
    scratch = public = reference = None
    serial = 0

    def area(self):
        type(self).serial += 1
        path = self.scratch / f'case-{self.serial}'
        path.mkdir(exist_ok=False)
        return path

    def cli(self, out, kind, seeds=None, reference=None):
        command = [sys.executable, '-B', '-m', 'learning.bc_crop.prepare', '--out', str(out),
                   '--public', str(self.public), '--kind', kind,
                   '--reference-registration', str(reference or self.reference)]
        if seeds is not None:
            command += ['--seed-config', str(seeds)]
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)

    def splits(self):
        return {key: read(self.reference)[key] for key in SEED_KEYS}

    def test_seed_contract(self):
        good = self.splits()
        self.assertEqual(seed_config(good), good)
        bad = []
        for name in good:
            value = deepcopy(good); value.pop(name); bad.append(value)
        value = deepcopy(good); value['extra'] = 1; bad.append(value)
        for replacement in (True, 1.5, '42', -1, 2**32, None):
            value = deepcopy(good); value['train_seeds'][0] = replacement; bad.append(value)
        for key in ('train_seeds', 'validation_seeds', 'test_seeds'):
            value = deepcopy(good); value[key] = value[key][:-1]; bad.append(value)
            value = deepcopy(good); value[key] = {}; bad.append(value)
        for key in ('train_seeds', 'validation_seeds', 'test_seeds'):
            value = deepcopy(good); value[key][-1] = good['train_seeds'][0]; bad.append(value)
        value = deepcopy(good); value['parity_seed'] = good['test_seeds'][0]; bad.append(value)
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                seed_config(value)

    def test_novelty_public_and_local_only_and_no_substring_collision(self):
        area = self.area(); public = area/'public'; history = area/'history'
        public.mkdir(); history.mkdir()
        splits = self.splits(); target = splits['test_seeds'][0]
        (public/'evidence.md').write_text(str(target)+'0', encoding='utf-8')
        self.assertTrue(novelty(splits, public, ['evidence.md'], history)['passed'])
        (public/'evidence.md').write_text(str(target), encoding='utf-8')
        with self.assertRaises(ValueError): novelty(splits, public, ['evidence.md'], history)
        (public/'evidence.md').write_text('', encoding='utf-8')
        save(history/'result.json', {'seed': target})
        with self.assertRaises(ValueError): novelty(splits, public, ['evidence.md'], history)

    def test_old_seeds_rejected_without_output_or_default_seed_fallback(self):
        area = self.area(); config = area/'input.json'; save(config, self.splits())
        out = area/'rejected'
        result = self.cli(out, 'new-holdout', config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Previously recorded seed', result.stderr)
        self.assertFalse(out.exists())
        result = self.cli(out, 'new-holdout')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(out.exists())
        config.write_bytes(b'')
        result = self.cli(out, 'new-holdout', config)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('Previously recorded seed', result.stderr)
        self.assertFalse(out.exists())

    def test_historical_cli_and_existing_destination(self):
        area = self.area(); out = area/'historical'
        result = self.cli(out, 'historical-replay')
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = integrity(out); original = read(self.reference)
        self.assertEqual({k: cfg[k] for k in SEED_KEYS}, self.splits())
        for key in ('training', 'bootstrap', 'engine_sha256', 'source_sha256', 'opponents', 'scope'):
            self.assertEqual(cfg[key], original[key])
        self.assertFalse(cfg['new_independent_evidence'])
        self.assertEqual(cfg['evaluation_role'], 'historical-replay')
        self.assertEqual((out/'reference-registration.json').read_bytes(), self.reference.read_bytes())
        before = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
        self.assertNotEqual(self.cli(out, 'historical-replay').returncode, 0)
        self.assertEqual(before, {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()})
        config = area/'input.json'; save(config, self.splits())
        self.assertNotEqual(self.cli(area/'bad', 'historical-replay', config).returncode, 0)
        self.assertFalse((area/'bad').exists())

    def test_historical_reference_cannot_be_promoted_to_new_holdout(self):
        area = self.area(); history = area/'historical'
        result = self.cli(history, 'historical-replay')
        self.assertEqual(result.returncode, 0, result.stderr)
        config = area/'input.json'; save(config, self.splits())
        out = area/'new'
        result = self.cli(out, 'new-holdout', config, history/'registration.json')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cannot supply new-holdout acceptance criteria', result.stderr)
        self.assertFalse(out.exists())

    def test_reference_identity_mismatch_rejected_before_output(self):
        area = self.area(); reference = area/'reference.json'
        for key in ('engine_sha256', 'teacher_sha256', 'source_sha256'):
            cfg = read(self.reference)
            cfg[key] = {} if key == 'source_sha256' else 'wrong'
            save(reference, cfg)
            out = area/key
            result = self.cli(out, 'historical-replay', reference=reference)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists())

    def test_new_holdout_cli_preserves_explicit_splits(self):
        area = self.area(); config = area/'input.json'
        # Disposable preparation fixture, not an experimental panel. Search a
        # disjoint range deterministically; do not run any policies on these seeds.
        base = 1_700_000_000
        splits = {'train_seeds': list(range(base, base+8)), 'validation_seeds': list(range(base+8, base+10)),
                  'test_seeds': list(range(base+10, base+18)), 'parity_seed': base+18}
        save(config, splits)
        out = area/'new'
        result = self.cli(out, 'new-holdout', config)
        self.assertEqual(result.returncode, 0, result.stderr)
        cfg = integrity(out)
        self.assertEqual({k: cfg[k] for k in SEED_KEYS}, splits)
        self.assertTrue(cfg['new_independent_evidence'])
        self.assertEqual(cfg['seed_config_sha256'], sha(config))
        self.assertGreater(cfg['seed_novelty']['public_text_files'], 0)
        self.assertGreater(cfg['seed_novelty']['local_evidence_files'], 0)

    def test_report_real_saved_metrics_historical_and_legacy(self):
        area = self.area(); out = area/'report'
        result = self.cli(out, 'historical-replay')
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = self.public/'review/evidence/bc-crop-v1-20261023'
        for name in ('test-ledger.json', 'audit-test.json', 'audit-entry-test.json'):
            (out/name).write_bytes((evidence/name).read_bytes())
        (out/'model').mkdir()
        (out/'model/report.json').write_bytes((evidence/'training-report.json').read_bytes())
        command = [sys.executable, '-B', '-m', 'learning.bc_crop.report', '--out', str(out)]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = read(out/'summary.json')
        self.assertIsNone(report['imitation_B_passed'])
        self.assertFalse(report['new_independent_evidence'])
        self.assertEqual(report['independent_seed_clusters'], 0)
        self.assertEqual(report['historical_seed_clusters'], 8)
        self.assertTrue(report['historical_imitation_metric_gate_passed'])
        self.assertIsNone(json.loads(run.stdout)['imitation_B_passed'])
        expected = read(evidence/'summary.json')
        for key in ('trained_minus', 'by_opponent', 'control', 'engineering_A_passed'):
            self.assertEqual(report[key], expected[key])
        (out/'registration.json').write_bytes(self.reference.read_bytes())
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(read(out/'summary.json'), expected)
        self.assertEqual(read(out/'paired-results.json'), read(evidence/'paired-results.json'))

    def test_report_role_contract(self):
        result = {'imitation_B_passed': True, 'independent_seed_clusters': 8, 'limitations': []}
        cfg = {'preparation_schema': 2, 'evaluation_role': 'new-holdout', 'new_independent_evidence': True}
        self.assertTrue(interpretation(result, cfg)['imitation_B_passed'])
        with self.assertRaises(ValueError):
            interpretation(result, {'new_independent_evidence': False})
        for key, value in [('new_independent_evidence', False), ('new_independent_evidence', 1),
                           ('evaluation_role', 'unknown'), ('preparation_schema', 3)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                interpretation(result, {**cfg, key: value})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--reference-registration', type=Path, required=True)
    args = parser.parse_args()
    if not args.scratch.is_dir() or any(args.scratch.iterdir()):
        parser.error('--scratch must be an existing empty directory inside a caller-owned temporary run')
    Preparation.scratch = args.scratch.resolve()
    Preparation.public = args.public.resolve()
    Preparation.reference = args.reference_registration.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Preparation))
    print(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun,
                      'failures': len(result.failures), 'errors': len(result.errors),
                      'training_runs': 0, 'new_games': 0}))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
