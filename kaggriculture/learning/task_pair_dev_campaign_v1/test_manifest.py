"""Pure synthetic manifest checks; no models, paths or private data read."""
from copy import deepcopy
import unittest

from .manifest import PrivateManifest, build_manifest


def inputs():
    return dict(
        seeds={'S03': 1002, 'S02': 1001},
        models={'baseline': dict(path='artifacts/synthetic/baseline.npz', sha256='a' * 64),
                'control': dict(path='artifacts/synthetic/control.npz', sha256='b' * 64),
                'auxiliary': dict(path='artifacts/synthetic/auxiliary.npz', sha256='c' * 64)},
        crop=dict(path='artifacts/synthetic/crop.npz', sha256='d' * 64),
        opponents={'market': dict(path='bots/synthetic_market.py', sha256='e' * 64),
                   'round2': dict(path='artifacts/synthetic/round2.py', sha256='f' * 64)},
        worker_sha256='0' * 64)


class ManifestTests(unittest.TestCase):
    def test_panel_roles_are_ordered_and_snapshotted(self):
        values = inputs(); manifest = build_manifest(**values)
        plans = manifest.panel.plans()
        self.assertEqual(len(plans), 24)
        self.assertTrue(all(row['seed'] == 1001 for row in plans if row['seed_alias'] == 'S02'))
        self.assertTrue(all(row['seed'] == 1002 for row in plans if row['seed_alias'] == 'S03'))
        self.assertEqual(manifest.opponent('market'), ('bots/synthetic_market.py', 'e' * 64))
        self.assertEqual(manifest.panel.worker_config()['source_sha256']['round2.py'], 'f' * 64)
        values['seeds']['S02'] = 777
        values['models']['control']['path'] = 'artifacts/elsewhere.npz'
        values['opponents']['market']['sha256'] = '1' * 64
        self.assertEqual(manifest.panel.seeds, (1001, 1002))
        self.assertEqual(manifest.panel.models[1], ('artifacts/synthetic/control.npz', 'b' * 64))
        self.assertEqual(manifest.opponent('market')[1], 'e' * 64)
        with self.assertRaises(AttributeError): manifest.panel = None

    def test_repr_and_errors_do_not_expose_values(self):
        values = inputs(); manifest = build_manifest(**values)
        output = repr(manifest) + repr(manifest.panel) + str(manifest)
        for value in ('1001', '1002', 'synthetic', 'a' * 64, 'e' * 64):
            self.assertNotIn(value, output)
        bad = inputs(); bad['opponents']['market']['path'] = '../secret.py'
        with self.assertRaises(ValueError) as raised: build_manifest(**bad)
        self.assertNotIn('../secret.py', str(raised.exception))

    def test_missing_extra_duplicate_roles_and_seed_types_rejected(self):
        variants = []
        for key in ('seeds', 'models', 'opponents'):
            missing = inputs(); del missing[key][next(iter(missing[key]))]; variants.append(missing)
            extra = inputs(); extra[key]['unexpected'] = 1; variants.append(extra)
            duplicate_pairs = inputs(); duplicate_pairs[key] = [('S02', 1), ('S02', 2)]; variants.append(duplicate_pairs)
        for value in (True, False, -1, 2**32, 1., '1'):
            bad = inputs(); bad['seeds']['S02'] = value; variants.append(bad)
        equal = inputs(); equal['seeds']['S03'] = equal['seeds']['S02']; variants.append(equal)
        for item in variants:
            with self.assertRaises(ValueError): build_manifest(**item)

    def test_rejects_noncanonical_traversal_and_windows_aliases(self):
        source_paths = ('../secret.py', 'bots/../secret.py', '/absolute/file.py',
                        'C' + ':' + '/private/file.py', 'bots\\file.py', 'bots//file.py',
                        'bots/./file.py', 'bots/dir /file.py', 'bots/NUL.py',
                        'bots/file.txt', 'bots/a\n.py')
        for path in source_paths:
            bad = inputs(); bad['opponents']['market']['path'] = path
            with self.subTest(kind='source'):
                with self.assertRaises(ValueError): build_manifest(**bad)
        model_paths = ('../secret.npz', 'artifacts/../escape.npz', '/a.npz',
                       'artifacts/NUL.npz', 'artifacts/dir /file.npz',
                       'artifacts/nested\\file.npz', 'artifacts/file.py')
        for path in model_paths:
            bad = inputs(); bad['models']['baseline']['path'] = path
            with self.subTest(kind='model'):
                with self.assertRaises(ValueError): build_manifest(**bad)

    def test_digest_and_identity_collisions_rejected(self):
        variants = []
        for bad_digest in ('A' * 64, 'g' * 64, 'a' * 63, True, None):
            bad = inputs(); bad['worker_sha256'] = bad_digest; variants.append(bad)
            bad = inputs(); bad['opponents']['market']['sha256'] = bad_digest; variants.append(bad)
            bad = inputs(); bad['models']['baseline']['sha256'] = bad_digest; variants.append(bad)
            bad = inputs(); bad['crop']['sha256'] = bad_digest; variants.append(bad)
        bad = inputs(); bad['opponents']['round2']['path'] = 'BOTS/SYNTHETIC_MARKET.py'; variants.append(bad)
        bad = inputs(); bad['opponents']['round2']['sha256'] = bad['opponents']['market']['sha256']; variants.append(bad)
        bad = inputs(); bad['models']['control'] = deepcopy(bad['models']['baseline']); variants.append(bad)
        bad = inputs(); bad['crop']['path'] = bad['models']['baseline']['path']; variants.append(bad)
        bad = inputs(); bad['crop']['sha256'] = bad['models']['baseline']['sha256']; variants.append(bad)
        for item in variants:
            with self.assertRaises(ValueError): build_manifest(**item)

    def test_direct_constructor_cannot_swap_opponent_bindings(self):
        manifest = build_manifest(**inputs())
        with self.assertRaises(ValueError): PrivateManifest(manifest.panel, tuple(reversed(manifest.opponent_sources)))
        with self.assertRaises(ValueError): manifest.opponent('unregistered')


if __name__ == '__main__':
    unittest.main()
