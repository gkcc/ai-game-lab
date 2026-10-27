import io
import unittest
import numpy as np
from learning.task_completion_bc_v1.numeric import pack, unpack, read, exploratory_gate
from learning.task_choice_v1.features import OPS


def group(n, seed=1):
    return dict(shared=np.zeros(275, np.float32), candidates=np.ones((n, 98), np.float32), chosen=n-1,
                candidate_ops=[OPS[i % len(OPS)] for i in range(n)], seed=seed, status='unknown')


def summary(model=.6, baseline=.5, count=30):
    return dict(count=count, model_top1=model, baseline_top1=baseline, model_ce=.5, baseline_ce=.6,
                model_operation_accuracy=model, baseline_operation_accuracy=baseline)


def metrics():
    return dict(overall=summary(), by_seed={'1': summary(), '2': summary()}, multi_candidate=summary(),
                cross_operation=summary(), by_operation={op: summary() for op in OPS})


class NumericTests(unittest.TestCase):
    def test_roundtrip_and_seed_is_metadata(self):
        a = pack([group(1), group(3, 4294967295)], 2, 4)
        f = io.BytesIO(); np.savez(f, **a); f.seek(0)
        b = read(f, 2, 4); out = unpack(b, 2, 4)
        self.assertEqual([x['chosen'] for x in out], [0, 2])
        self.assertEqual(out[1]['candidate_ops'], list(OPS[:3]))
        changed = pack([group(1, 99), group(3, 7)], 2, 4)
        np.testing.assert_array_equal(a['shared'], changed['shared'])
        np.testing.assert_array_equal(a['candidates'], changed['candidates'])
        with self.assertRaises(ValueError): unpack(a, 2, 3)
        self.assertEqual(unpack(pack([], 0, 0), 0, 0), [])

    def test_tamper(self):
        a = pack([group(2)], 1, 2)
        for key, value in [('offsets', np.array([0, 1], np.int64)), ('chosen', np.array([2], np.int64)),
                           ('status', np.array([2], np.uint8)), ('ops', np.array([0, 99], np.uint8)),
                           ('seed', np.array([1], np.int64))]:
            b = dict(a); b[key] = value
            with self.assertRaises(ValueError): unpack(b, 1, 2)

    def test_gate(self):
        m = metrics(); self.assertTrue(exploratory_gate(m, [1, 2])['passed'])
        for section, field, value in [('overall', 'model_top1', .54), ('overall', 'model_ce', .7),
                                      ('multi_candidate', 'model_top1', .4), ('cross_operation', 'count', 0)]:
            m = metrics(); m[section][field] = value
            self.assertFalse(exploratory_gate(m, [1, 2])['passed'])
        m = metrics(); m['by_seed']['2']['model_top1'] = .5
        self.assertFalse(exploratory_gate(m, [1, 2])['passed'])
        m = metrics()
        for op in OPS[2:]: m['by_operation'][op]['count'] = 19
        self.assertFalse(exploratory_gate(m, [1, 2])['passed'])
        m = metrics()
        for op in OPS: m['by_operation'][op]['model_operation_accuracy'] = .54
        self.assertFalse(exploratory_gate(m, [1, 2])['passed'])
        m = metrics(); m['overall']['model_ce'] = float('nan')
        self.assertFalse(exploratory_gate(m, [1, 2])['passed'])
        self.assertFalse(exploratory_gate({}, [1])['passed'])

if __name__ == '__main__': unittest.main()
