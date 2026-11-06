"""Pure checks preventing partial or type-confused eight-step success receipts."""
from copy import deepcopy
import unittest

from . import contract, fixtures, runner


def accepted():
    counts = {k+suffix:v for k,v in contract.PROBE_CAPS.items()
              for suffix in ('_attempted', '_returned')}
    counts.update(direct_boundary_requests=2, evidence_write_failures=0)
    pairs = [(n,o) for n in fixtures.NAMES for o in fixtures.OPS]
    result = dict(passed=True, counts=deepcopy(counts), model_forwards=0, complete_games=0,
                  rows=[dict(scene=n, op=o) for n,o in pairs])
    emitted = [dict(name=n+'-'+o.lower()+'.local.json', sha256='synthetic') for n,o in pairs]
    return result, counts, emitted


class ResultContract(unittest.TestCase):
    def test_exact_eight_closed_branches_pass(self):
        self.assertTrue(runner.validate_result(*accepted(), contract, fixtures))

    def test_partial_or_type_confused_counts_reject(self):
        for key, value in (('runs_returned',True), ('action_steps_returned',8.0),
                           ('action_steps_returned',7), ('evidence_write_failures',1)):
            with self.subTest(key=key, value=value):
                result, counts, emitted = accepted()
                counts[key] = value
                result['counts'] = deepcopy(counts)
                with self.assertRaises(ValueError):
                    runner.validate_result(result, counts, emitted, contract, fixtures)

    def test_missing_repeated_or_misordered_branches_reject(self):
        for mode in ('missing', 'repeated', 'misordered'):
            result, counts, emitted = accepted()
            if mode == 'missing':
                result['rows'].pop()
            elif mode == 'repeated':
                result['rows'][-1] = result['rows'][0]
            else:
                emitted.reverse()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                runner.validate_result(result, counts, emitted, contract, fixtures)


if __name__ == '__main__':
    unittest.main()
