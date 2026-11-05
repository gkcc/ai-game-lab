"""Small public reporting contract: no models, private files or engine calls."""
from copy import deepcopy
import unittest

from .contract import project, sources


def rows():
    result = []
    for source, own in zip(sources(), (10, -6, 0, 3)):
        result.append({**{k:source[k] for k in ('role','opponent','side','purpose')},
            'status':'paired','technical_passed':True,
            'request_changed':True,'physical_changed':True,
            'baseline':{'start':{'own':100,'opponent':200},'end':{'own':120,'opponent':205}},
            'intervention':{'start':{'own':100,'opponent':200},'end':{'own':120+own,'opponent':205}}})
    return result


class ContractTests(unittest.TestCase):
    def test_cash_signal_is_not_a_final_match_score_or_training_promotion(self):
        result = project(rows())
        self.assertTrue(result['minimum_label_signal'])
        self.assertEqual(result['paired_only_mean_delta'], {'own':1.75,'opponent':0.,'margin':1.75})
        self.assertEqual(result['positive_zero_negative']['margin'], {'positive':2,'zero':1,'negative':1})
        self.assertEqual(result['cash_horizon_steps'],48)
        self.assertFalse(result['final_profit_evaluated'])
        self.assertFalse(result['terminal_score_evaluated'])
        self.assertFalse(result['training'])
        self.assertFalse(result['promotion'])

    def test_missing_is_not_zero_and_all_missing_has_no_mean(self):
        values=rows()
        for index in range(4):
            values[index] = {**{k:values[index][k] for k in ('role','opponent','side','purpose')},
                             'status':'no_eligible_target','target':None}
        result=project(values)
        self.assertEqual(result['paired_roles'],0)
        self.assertIsNone(result['paired_only_mean_delta'])
        self.assertEqual(result['positive_zero_negative']['own']['zero'],0)
        self.assertFalse(result['minimum_label_signal'])
        values[0]['baseline']=rows()[0]['baseline']
        with self.assertRaises(ValueError):project(values)

    def test_signal_requires_nonzero_diagnostic_role_and_both_margin_signs(self):
        values=rows()
        for index in (2,3):values[index]['intervention']=deepcopy(values[index]['baseline'])
        self.assertFalse(project(values)['minimum_label_signal'])
        values=rows();values[1]['intervention']['end']['own']=126
        self.assertFalse(project(values)['minimum_label_signal'])

    def test_relative_gain_from_hurting_opponent_only_does_not_meet_own_cash_gate(self):
        values=rows()
        for index in (0,3):
            values[index]['intervention']['end']={'own':119,'opponent':190}
        self.assertFalse(project(values)['minimum_label_signal'])

    def test_effect_count_and_physical_cash_consistency_are_required(self):
        values=rows()
        for index in (0,2):
            values[index]['intervention']=deepcopy(values[index]['baseline'])
            values[index]['physical_changed']=False
        self.assertFalse(project(values)['minimum_label_signal'])
        values[0]['intervention']['end']['own']+=1
        with self.assertRaises(ValueError):project(values)

    def test_technical_failure_missing_case_wrong_order_or_start_cash_is_not_reportable(self):
        original=rows()
        cases=[original[:-1],list(reversed(original))]
        failed=deepcopy(original);failed[1]['technical_passed']=False;cases.append(failed)
        wrong=deepcopy(original);wrong[0]['intervention']['start']['own']+=1;cases.append(wrong)
        boolean=deepcopy(original);boolean[0]['side']=False;cases.append(boolean)
        for value in cases:
            with self.subTest(value=value),self.assertRaises(ValueError):project(value)


if __name__=='__main__':
    unittest.main()
