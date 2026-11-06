"""Reject partial official execution and hidden extra model trajectories."""
from copy import deepcopy
import unittest

from . import contract,runner


def operations():
    caps={'live':dict(environment_constructions=1,resets=2,official_steps=719,actor_calls=719,rival_calls=719),
        'reconstruction':dict(actor_calls=719,rival_calls=719),
        'replay':dict(environment_constructions=1,resets=1,official_steps=719)}
    return {p:{k+s:v for k,v in items.items() for s in ('_attempted','_returned')} for p,items in caps.items()}


class ReceiptContracts(unittest.TestCase):
    def test_requires_all_three_actual_operation_phases(self):
        runner.operation_check(operations())
        for phase,key,value in [('live','resets_returned',1),('replay','official_steps_returned',718),
                                ('reconstruction','actor_calls_returned',True),('live','rival_calls_returned',719.)]:
            counts=operations();counts[phase][key]=value
            with self.subTest(phase=phase,key=key),self.assertRaises(ValueError):runner.operation_check(counts)

    def test_model_total_includes_precheck_and_exactly_two_complete_trajectories(self):
        cost=dict(task_forwards=1000,task_rows=3000,crop_forwards=600,crop_rows=600,
            gate_evaluations=0,controller_calls=1438,teacher_proposals=0)
        pre=dict(model_costs=dict(task_forwards=30,task_rows=100,crop_forwards=5,crop_rows=5,
            gate_evaluations=0,controller_calls=8,teacher_proposals=0))
        rows=[dict(model_counts=deepcopy(cost)) for _ in range(4)]
        actual=runner.aggregate_costs(pre,rows,contract)
        self.assertEqual(actual['controller_calls'],5760)
        self.assertEqual(actual['task_forwards'],4030)
        for altered in (1438+719,1438-719,1438.):
            bad=deepcopy(rows);bad[0]['model_counts']['controller_calls']=altered
            with self.subTest(altered=altered),self.assertRaises(ValueError):runner.aggregate_costs(pre,bad,contract)


if __name__=='__main__':unittest.main()
