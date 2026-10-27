import unittest
from .evaluation import plans,evaluate


def rows():
    result=[]
    for p in plans([11,22]):
        result.append(dict(**p,passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
            net_profit=10000,opponent_net_profit=9000,bank=11000,opponent_bank=10000,margin=1000,score=1.))
    return result


class EvaluationTest(unittest.TestCase):
    def test_fixed_panel_and_inclusive_guards(self):
        data=rows()
        for row in data:
            if row['policy']=='candidate':
                row['net_profit']-=1000; row['bank']-=1000; row['margin']-=1000; row['score']=.5
        result=evaluate(data,[11,22])
        self.assertTrue(result['exploratory_gate_passed'])
        self.assertEqual(result['seed_groups'][0]['mean_own_profit_difference'],-1000)
        self.assertEqual(result['pairs'][0]['candidate_outcome'],'draw')
        data[0]['net_profit']-=1001
        self.assertFalse(evaluate(data,[11,22])['exploratory_gate_passed'])

    def test_reject_partial_duplicate_and_bad_values(self):
        for bad in (rows()[:-1],rows()[:-1]+[rows()[0]]):
            with self.assertRaises(ValueError): evaluate(bad,[11,22])
        data=rows(); data[0]['net_profit']=float('nan')
        with self.assertRaises(ValueError): evaluate(data,[11,22])
        with self.assertRaises(ValueError): plans([11,11])
        data=rows(); data[0]['margin']=3
        with self.assertRaises(ValueError): evaluate(data,[11,22])


if __name__ == '__main__': unittest.main()
