import types
import unittest
from .experiment import OPS_COUNTS,expected_weights
from .weights import operation_weights


class RegistrationTest(unittest.TestCase):
    def test_registered_and_training_weight_formulas_agree(self):
        groups=[types.SimpleNamespace(teacher_choice_label=0,candidate_ops=(op,))
                for op,count in OPS_COUNTS.items() for _ in range(count)]
        self.assertEqual(expected_weights(OPS_COUNTS),operation_weights(groups))
        bad={**OPS_COUNTS,'DIG':OPS_COUNTS['DIG']+1}
        with self.assertRaises(ValueError): expected_weights(bad)


if __name__=='__main__': unittest.main()
