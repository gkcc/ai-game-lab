import json
import unittest

from .evaluation import GATES, evaluate, plans


SEEDS=[3112345671,3112345682,3112345693,3112345704]


def panel():
    return [dict(p,passed=True,states=720,action_turns=719,status=['DONE','DONE'],agent_errors=0,
                 initial_bank=10000,opponent_initial_bank=10000,bank=20000,opponent_bank=10000,
                 net_profit=10000,opponent_net_profit=0,margin=10000,score=1) for p in plans(SEEDS)]


def adjust(row, delta):
    row['bank']=20000+delta; row['net_profit']=10000+delta; row['margin']=10000+delta
    row['score']=1 if row['margin']>0 else .5 if row['margin']==0 else 0


def chosen(rows, alias, opponent, side, policy='merged'):
    return next(row for row in rows if (row['seed_alias'],row['opponent'],row['side'],row['policy'])==
                (alias,opponent,side,policy))


class EvaluationTests(unittest.TestCase):
    def test_alias_plan_and_seed_validation(self):
        actual=plans(SEEDS)
        self.assertEqual(len(actual),72)
        self.assertEqual(len({p['game'] for p in actual}),72)
        self.assertEqual({p['seed_alias'] for p in actual},{'S01','S02','S03','S04'})
        self.assertTrue(all(str(p['seed']) not in p['game'] for p in actual))
        self.assertEqual([p['game'] for p in actual],[p['game'] for p in plans([1,2,3,4])])
        for seeds in ([1,2,3], [1,2,3,3], [1,2,3,True], [1,2,3,-1], [1,2,3,2**32]):
            with self.assertRaises(ValueError): plans(seeds)

    def test_incomplete_duplicate_and_inconsistent_rows_fail(self):
        rows=panel()
        with self.assertRaises(ValueError): evaluate(rows[:-1],SEEDS)
        rows[-1]=rows[0].copy()
        with self.assertRaises(ValueError): evaluate(rows,SEEDS)
        for field,value in (('net_profit',2),('opponent_net_profit',1),('margin',2),('score',0),
                            ('bank',float('nan')),('bank',10**400),('seed',True),('side',True),('states',True)):
            rows=panel(); rows[0][field]=value
            with self.subTest(field=field,value=str(value)[:20]):
                with self.assertRaises(ValueError): evaluate(rows,SEEDS)
        rows=panel(); rows[0]['seed_alias']='S02'
        with self.assertRaises(ValueError): evaluate(rows,SEEDS)

    def test_pair_guard_boundary_and_failure(self):
        rows=panel(); a=chosen(rows,'S01','round2',0); b=chosen(rows,'S01','round2',1)
        adjust(a,-2000); adjust(b,2000)
        result=evaluate(rows,SEEDS)
        self.assertTrue(result['exploratory_gate_passed']); self.assertFalse(result['exploratory_positive_signal'])
        adjust(a,-2001); adjust(b,2001)
        result=evaluate(rows,SEEDS)
        self.assertFalse(result['exploratory_gate_passed'])
        self.assertTrue(any(f['scope']=='pair' and f['seed_alias']=='S01' and f['side']==0 for f in result['gate_failures']))

    def test_seed_opponent_guard_and_positive_signal(self):
        rows=panel(); a=chosen(rows,'S01','round2',0); b=chosen(rows,'S01','round2',1); offset=chosen(rows,'S01','market',0)
        adjust(a,-1000); adjust(b,-1000); adjust(offset,2000)
        self.assertTrue(evaluate(rows,SEEDS)['exploratory_gate_passed'])
        adjust(a,-1001); adjust(b,-1001); adjust(offset,2002)
        result=evaluate(rows,SEEDS)
        self.assertFalse(result['exploratory_gate_passed'])
        self.assertTrue(any(f['scope']=='seed_opponent' and f['opponent']=='round2' for f in result['gate_failures']))
        rows=panel()
        for row in rows:
            if row['policy']=='merged': adjust(row,100)
        result=evaluate(rows,SEEDS)
        self.assertTrue(result['exploratory_gate_passed']); self.assertTrue(result['exploratory_positive_signal'])

    def test_shared_seed_cluster_bootstrap_and_alias_only_result(self):
        rows=panel()
        for row in rows:
            delta=100*int(row['seed_alias'][1:])
            if row['policy']=='merged': adjust(row,delta)
            elif row['policy']=='teacher_task': adjust(row,-delta)
        first=evaluate(rows,SEEDS); second=evaluate(rows,SEEDS)
        self.assertEqual(first['bootstrap_intervals'],second['bootstrap_intervals'])
        self.assertEqual(first['gates'],GATES); self.assertEqual(first['seed_cluster_count'],4)
        self.assertEqual([c['own_profit_difference'] for c in first['seed_clusters']['unweighted']],[100,200,300,400])
        for metric in ('own_profit_difference','margin_difference'):
            base=first['bootstrap_intervals']['unweighted'][metric]
            teacher=first['bootstrap_intervals']['teacher_task'][metric]
            self.assertEqual(teacher,[2*x for x in base])
        self.assertEqual(len(first['summaries']),9)
        self.assertTrue(all(row['wins']==8 for row in first['summaries']))
        output=json.dumps(first)
        self.assertTrue(all(str(seed) not in output for seed in SEEDS))
        self.assertFalse(first['automatic_promotion'])


if __name__=='__main__': unittest.main()
