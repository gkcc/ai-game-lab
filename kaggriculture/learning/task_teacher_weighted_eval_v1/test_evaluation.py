import unittest
from .evaluation import plans, evaluate


SEEDS=[1,2,3]


def panel():
    return [dict(p,passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
                 initial_bank=10000,opponent_initial_bank=10000,bank=11000,opponent_bank=10000,
                 net_profit=1000,opponent_net_profit=0,margin=1000,score=1) for p in plans(SEEDS)]


def adjust(row, value):
    row.update(bank=11000+value,net_profit=1000+value,margin=1000+value,
               score=1 if 1000+value>0 else 0 if 1000+value<0 else .5)


class EvaluationTests(unittest.TestCase):
    def test_plan_validation(self):
        self.assertEqual(len(plans(SEEDS)),24)
        for seeds in ([1,1,2],[1,2,True],[1,2,-1],[1,2,2**32]):
            with self.assertRaises(ValueError): plans(seeds)

    def test_missing_and_duplicate(self):
        rows=panel()
        with self.assertRaises(ValueError): evaluate(rows[:-1],SEEDS)
        rows[-1]=rows[0].copy()
        with self.assertRaises(ValueError): evaluate(rows,SEEDS)

    def test_economics_and_outcome_rejected(self):
        for field,value in [('net_profit',2),('opponent_net_profit',1),('margin',2),('score',0),('bank',float('nan'))]:
            rows=panel(); rows[0][field]=value
            with self.assertRaises(ValueError): evaluate(rows,SEEDS)

    def test_boolean_seed_and_side_rejected(self):
        for field in ('seed','side'):
            rows=panel()
            row=next(r for r in rows if r[field]==1)
            row[field]=True
            with self.assertRaises(ValueError): evaluate(rows,SEEDS)

    def test_finite_inputs_with_overflowing_difference_rejected(self):
        rows=panel()
        for row in rows:
            if row['seed']==1 and row['opponent']=='round2' and row['side']==0:
                value=1e308 if row['policy']=='weighted' else -1e308
                row.update(bank=value,initial_bank=0,net_profit=value,opponent_bank=0,
                           opponent_initial_bank=0,opponent_net_profit=0,margin=value,
                           score=1 if value>0 else 0)
        with self.assertRaises(ValueError): evaluate(rows,SEEDS)

    def test_threshold_equality_and_failure(self):
        rows=panel()
        a=next(r for r in rows if r['seed']==1 and r['opponent']=='round2' and r['side']==0 and r['policy']=='weighted')
        adjust(a,-2000)
        self.assertTrue(evaluate(rows,SEEDS)['exploratory_gate_passed'])
        adjust(a,-2001)
        self.assertFalse(evaluate(rows,SEEDS)['exploratory_gate_passed'])

    def test_seed_clusters_and_reproducible_bootstrap(self):
        rows=panel()
        for r in rows:
            if r['policy']=='weighted': adjust(r,100*r['seed'])
        first=evaluate(rows,SEEDS); second=evaluate(rows,SEEDS)
        self.assertEqual([c['own_profit_difference'] for c in first['seed_clusters']],[100,200,300])
        self.assertEqual(first['bootstrap_intervals'],second['bootstrap_intervals'])
        self.assertEqual(first['bootstrap_replicates'],10000)
        self.assertEqual(first['bootstrap_intervals']['own_profit_difference'],[100,300])
        self.assertEqual(len(first['summaries']),4)
        self.assertTrue(all(s['wins']==6 for s in first['summaries']))


if __name__=='__main__': unittest.main()
