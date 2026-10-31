"""Synthetic replay/trace comparisons with no official engine execution."""
from copy import deepcopy
from dataclasses import replace
import io,json,unittest

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_student_dagger_collection_v1.sources import _TraceReader
from . import checks


def game():
    steps=[]
    for step in range(720):
        banks=[{'money':3000},{'money':3000}]
        frame=[]
        for side in (0,1):
            frame.append(dict(observation=dict(farms=deepcopy(banks),
                private=dict(inventories=[{'A':1,'B':2}],shed={'A':1,'B':2})),
                action=dict(farmer=['PASS'],hands=[],market=[]),
                status='DONE' if step==719 else 'ACTIVE',reward=3000 if step==719 else None))
        steps.append(frame)
    return dict(configuration={'seed':None},info={'seed':17},steps=steps)


def trace(rows): return io.StringIO(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in rows))


class PairChecks(unittest.TestCase):
    def test_account_rejects_different_initial_cash_and_nonfinite_money(self):
        a=checks.compact(game(),target_step=0,guard=lambda:None)
        with self.assertRaises(ValueError):
            checks.account(a,replace(a,initial_cash=(3001,3000)),side=0)
        with self.assertRaises(ValueError):
            checks.account(a,replace(a,final_cash=(float('nan'),3000)),side=0)
        b=replace(a,final_cash=(3300,2900))
        for side in (0,1):
            delta=checks.account(a,b,side=side)['delta']
            self.assertEqual(delta['margin'],delta['own']-delta['opponent'])

    def test_same_source_then_target_change_for_both_seats_and_terminal_step(self):
        for step,side in ((0,0),(70,1),(718,0)):
            a=game();b=deepcopy(a)
            b['steps'][step+1][side]['action']['farmer']=['DIG']
            b['steps'][step+1][side]['observation']['private']['shed']['A']+=1
            ca=checks.compact(a,target_step=step,guard=lambda:None)
            cb=checks.compact(b,target_step=step,guard=lambda:None)
            self.assertTrue(checks.control_matches_history(ca,ca))
            self.assertTrue(checks.intervention_matches_prefix(ca,cb,target_step=step,side=side))
            self.assertEqual(checks.account(ca,cb,side=side)['delta']['margin'],0)
            self.assertNotIn('17',repr(ca))

    def test_rejects_changed_shed_opponent_or_missing_nonaction_effect(self):
        a=game();ca=checks.compact(a,target_step=1,guard=lambda:None)
        for variant in ('shed','opponent','no_effect'):
            b=deepcopy(a)
            b['steps'][2][0]['action']['farmer']=['DIG']
            b['steps'][2][0]['observation']['private']['shed']['A']+=1
            if variant=='shed': b['steps'][0][1]['observation']['private']['shed']={'B':2,'A':1}
            if variant=='opponent': b['steps'][2][1]['action']['farmer']=['WATER']
            if variant=='no_effect': b['steps'][2][0]['observation']['private']['shed']['A']-=1
            cb=checks.compact(b,target_step=1,guard=lambda:None)
            with self.subTest(variant=variant),self.assertRaises(ValueError):
                checks.intervention_matches_prefix(ca,cb,target_step=1,side=0)

    def test_trace_comparison_uses_dynamic_target_and_respects_prior_choices(self):
        step,ordinal=2,1
        value=dict(observation=dict(step=step),candidates=[dict(id='student'),dict(id='teacher')])
        expected=dict(input_sha256=ordered_hash(value),original_choice='student',teacher_choice='teacher')
        record=dict(inputs=value,chosen='student',teacher_label='teacher',source='callback')
        rows=[dict(step=i,records=[deepcopy(record),deepcopy(record)] if i==step else [],crop_records=[])
              for i in range(719)]
        changed=deepcopy(rows);changed[step]['records'][ordinal]['chosen']='teacher'
        result=checks.control_trace(trace(rows),trace(rows),target_step=step,ordinal=ordinal,
                                    expected=expected,guard=lambda:None)
        self.assertEqual(result['chosen'],'student')
        # The authenticated historical reader rejects unbounded reads and
        # requires exactly 719 bounded lines followed by a one-byte EOF read.
        historical=_TraceReader(io.BytesIO(trace(rows).getvalue().encode()),lambda:None)
        checks.control_trace(historical,trace(rows),target_step=step,ordinal=ordinal,
                             expected=expected,guard=lambda:None)
        self.assertTrue(historical._complete)
        self.assertTrue(checks.intervention_trace(trace(rows),trace(changed),target_step=step,
            ordinal=ordinal,expected=expected,crop_before=0,guard=lambda:None))
        changed[step]['records'][0]['chosen']='teacher'
        with self.assertRaises(ValueError): checks.intervention_trace(trace(rows),trace(changed),
            target_step=step,ordinal=ordinal,expected=expected,crop_before=0,guard=lambda:None)


if __name__=='__main__': unittest.main()
