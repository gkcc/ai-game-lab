"""Bounded coordinator seams only: synthetic metadata, no files or execution."""
from contextlib import ExitStack
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from learning.task_bc_v1 import model
from learning.task_choice_v1.features import OPS
from learning.task_harvest_model_dev_v1 import runner as frozen_helpers
from . import contract,runner,sources


KEYS=('task_forwards','task_rows','crop_forwards','crop_rows','gate_evaluations','controller_calls','teacher_proposals')
TASK='a'*64


def fixture(*,missing=()):
    documents={};cases=[];references=[]
    h=dict(own_net_profit=10,opponent_net_profit=0,margin=10,score=1.)
    w=dict(own_net_profit=0,opponent_net_profit=0,margin=0,score=.5)
    for index,identity in enumerate(contract.sources()):
        origin=identity['role']
        source=dict(identity,row=dict(seed=100+index),task=['artifacts/synthetic/task.npz',TASK])
        cases.append(source)
        points=[]
        for q,step in zip(contract.POINTS,(240,360,600)):
            absent=origin+'-'+q in missing
            points.append(dict(label=q,index=int(q[1:]),reason='duplicate_index' if absent else None,
                target=None if absent else dict(step=step,side=source['side'],synthetic_point=q)))
        documents[origin]=dict(origin=origin,source=deepcopy(source),
            quantiles=dict(points=points,counts=dict(trace_lines=719)),trace_index=[],trace_scans=[])
        references.append(dict(origin=origin,row=dict(identity,status='paired',technical_passed=True,H=deepcopy(h),W=deepcopy(w))))
    archive=dict(cases=cases,references=references)
    plans=sources.plans([documents[o] for o in contract.ORIGINS]);records=[]
    for plan in plans:
        source=next(c for c in cases if c['role']==plan['origin']);active=plan['target_sha256'] is not None
        cost={k:0 for k in KEYS}
        row={k:source[k] for k in ('role','cohort','opponent','side')}
        if active:
            cost.update(controller_calls=1438,task_forwards=100,crop_forwards=100,task_rows=100,crop_rows=100)
            row.update(status='paired',technical_passed=True,H=deepcopy(h),W=deepcopy(w))
        else:row.update(status='no_eligible_target')
        records.append(dict(plan=plan,row=row,model_costs=cost,official_actions=plan['official_actions']))
    return archive,documents,plans,records


class NormalizationFixture:
    """Only the caller seam is stubbed; analysis algorithms have their own tests."""
    def __init__(self,case,*,analysis_error=None,closure_error=False,budget_error=None):
        self.archive,self.documents,_plans,self.records=fixture(missing=contract.ROLES)
        names=['local-'+str(i) for i in range(98)];shared=['shared-'+str(i) for i in range(275)]
        inputs=dict(schema_version='synthetic-schema',schema_sha256='b'*64,
            feature_names=names,shared_feature_names=shared)
        references=[dict(origin=o,point='Q0',sample=dict(inputs=deepcopy(inputs))) for o in contract.ORIGINS]
        metadata=dict(inputs,dimensions=list(model.DIMENSIONS),ops=list(OPS))
        self.arrays=dict(metadata=np.asarray(json.dumps(metadata)),mean=np.zeros(373),scale=np.ones(373))
        class NPZ:
            def __enter__(_self):return self.arrays
            def __exit__(_self,*_active):return False
        self.load=Mock(return_value=NPZ());self.writes=[]
        self.analyzer=Mock(side_effect=analysis_error) if analysis_error else Mock(return_value=dict(
            public=dict(counts=dict(encoded_hw_rows=8,model_forwards=0,model_constructions=0)),private=dict(synthetic=True)))
        self.dm=SimpleNamespace(MAX_MODEL=1024,_hash=Mock(return_value=TASK),
            _read=Mock(side_effect=AssertionError('No new samples or historical files may be read')))
        self.modules=dict(driver=self.dm,
            source=SimpleNamespace(reference_samples=Mock(return_value=references)),
            analysis=SimpleNamespace(analyze=self.analyzer))
        self.infra=SimpleNamespace(decode=json.loads)
        def write(_infra,name,value,*,closure=False):
            self.writes.append((name,deepcopy(value)))
            if closure and closure_error:raise BufferError('synthetic closure failure')
            return 'c'*64
        self.stack=ExitStack();case.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runner,'shared',return_value=frozen_helpers))
        self.stack.enter_context(patch.object(frozen_helpers,'private_write',side_effect=write))
        self.stack.enter_context(patch('numpy.load',self.load))
        self.stack.enter_context(patch.object(model.TaskNetwork,'__init__',side_effect=AssertionError('No model construction')))
        self.stack.enter_context(patch.object(model.TaskNetwork,'score',side_effect=AssertionError('No score call')))
        if budget_error:
            self.stack.enter_context(patch.object(runner,'aggregate_costs',side_effect=budget_error))

    def run(self):
        return runner.analyze_completed(self.infra,self.modules,self.archive,self.documents,self.records,
            contract,lambda:True)

    def closed(self):
        return next(value for name,value in self.writes if name=='analysis-closed.local.json')


class RunnerTests(unittest.TestCase):
    def test_twelve_unique_new_jobs_keep_original_source_roles_and_exclude_q0(self):
        archive,documents,plans,_records=fixture();before=deepcopy(archive)
        self.assertEqual([p['role'] for p in plans],list(contract.ROLES))
        self.assertEqual(len({p['role'] for p in plans}),12)
        self.assertEqual({p['point'] for p in plans},{'Q1','Q2','Q3'})
        for plan in plans:
            source,selected=runner.point_selection(sources,archive,documents,plan)
            self.assertEqual(source['role'],plan['origin'])
            self.assertEqual(selected['source']['role'],plan['origin'])
            self.assertEqual(selected['role'],plan['role'])
            self.assertEqual(selected['point'],plan['point'])
            for kind in ('permit','started','consumed'):
                self.assertEqual(runner.marker(kind,'d'*40,'e'*64,plan['role'])['role'],plan['role'])
        self.assertEqual(archive,before)
        for kind in ('permit','started','consumed'):
            with self.assertRaises(ValueError):runner.marker(kind,'d'*40,'e'*64,'F01-Q0')
        with self.assertRaises(ValueError):sources.selected_point(documents['F01'],'Q0')

    def test_point_binding_rejects_other_point_source_or_changed_target(self):
        archive,documents,plans,_records=fixture()
        for defect in ('point','target','source','actions'):
            with self.subTest(defect=defect):
                plan=deepcopy(plans[0]);docs=deepcopy(documents)
                if defect=='point':plan['point']='Q2'
                elif defect=='target':docs['F01']['quantiles']['points'][0]['target']['step']+=1
                elif defect=='source':docs['F01']['source']['row']['seed']+=1
                else:plan['official_actions']+=2
                with self.assertRaises(ValueError):runner.point_selection(sources,archive,docs,plan)

    def test_aggregate_includes_diagnostic_rows_in_total_cap_and_binds_steps(self):
        _archive,_documents,_plans,records=fixture()
        result=runner.aggregate_costs(records,contract,32)
        self.assertEqual(result['model_costs']['controller_calls'],17256)
        self.assertEqual(result['rival_calls'],17256)
        self.assertEqual((result['paired'],result['terminal_suffixes']),(12,24))
        self.assertEqual(result['official_actions'],sum(r['plan']['official_actions'] for r in records))
        changed=deepcopy(records);changed[0]['official_actions']+=2
        with self.assertRaises(ValueError):runner.aggregate_costs(changed,contract,32)
        changed=deepcopy(records);changed[0]['model_costs']['controller_calls']-=1
        with self.assertRaises(ValueError):runner.aggregate_costs(changed,contract,32)
        for record in records:
            record['model_costs']['task_rows']=320000;record['model_costs']['crop_rows']=0
        self.assertEqual(runner.aggregate_costs(records,contract,0)['total_input_rows_including_analysis'],3840000)
        with self.assertRaises(ValueError):runner.aggregate_costs(records,contract,32)
        records[-1]['model_costs']['task_rows']-=32
        self.assertEqual(runner.aggregate_costs(records,contract,32)['total_input_rows_including_analysis'],3840000)
        with self.assertRaises(ValueError):runner.aggregate_costs(records,contract,33)

    def test_ledger_reuses_four_references_and_keeps_missing_returns_null(self):
        archive,documents,plans,records=fixture(missing=('F02-Q2',));before=deepcopy((archive,records))
        ledger=runner.ledger_rows(archive,records,contract);public=contract.projection(ledger)
        self.assertEqual(len(ledger),16)
        self.assertEqual(public['reused_pairs'],4)
        self.assertEqual(public['completed_new_pairs'],11)
        self.assertEqual(sum(row['reused'] for row in public['rows']),4)
        for i,reference in enumerate(archive['references']):
            self.assertEqual(ledger[4*i]['H'],reference['row']['H'])
            self.assertEqual(ledger[4*i]['W'],reference['row']['W'])
        absent=next(row for row in public['rows'] if (row['origin'],row['point'])==('F02','Q2'))
        self.assertIsNone(absent['returns'])
        self.assertEqual(absent['reason'],'duplicate_index')
        self.assertFalse(absent['reused'])
        self.assertFalse(any(plan['point']=='Q0' for plan in plans))
        self.assertEqual((archive,records),before)
        self.assertEqual(runner.aggregate_costs(records,contract,30)['model_costs']['controller_calls'],11*1438)

    def test_normalization_is_loaded_once_without_model_and_rows_are_reserved(self):
        probe=NormalizationFixture(self)
        public,costs,meter=probe.run()
        probe.load.assert_called_once_with(runner.GAME/'artifacts/synthetic/task.npz',allow_pickle=False)
        self.assertEqual(meter['normalization_loads_attempted'],1)
        self.assertEqual(meter['normalization_loads_returned'],1)
        self.assertEqual(meter['feature_rows_requested'],8)
        self.assertEqual(meter['feature_rows_completed'],8)
        self.assertEqual(meter['model_forwards'],0)
        self.assertEqual(meter['model_constructions'],0)
        self.assertEqual(costs['analysis_feature_rows'],8)
        self.assertEqual(costs['total_input_rows_including_analysis'],8)
        self.assertEqual(public['counts']['encoded_hw_rows'],8)
        probe.dm._read.assert_not_called()
        self.assertEqual(probe.closed()['outcome'],'completed')
        self.assertFalse(probe.closed()['feature_rows_actual_unknown'])

    def test_failed_analysis_keeps_unknown_rows_and_original_error_when_closing_fails(self):
        primary=ValueError('synthetic diagnostic failure')
        probe=NormalizationFixture(self,analysis_error=primary,closure_error=True)
        with self.assertRaises(ValueError) as caught:probe.run()
        self.assertIs(caught.exception,primary)
        self.assertEqual(probe.load.call_count,1)
        closed=probe.closed()
        self.assertEqual(closed['outcome'],'failed')
        self.assertEqual(closed['meter']['feature_rows_requested'],8)
        self.assertIsNone(closed['meter']['feature_rows_completed'])
        self.assertTrue(closed['feature_rows_actual_unknown'])
        self.assertNotIn('feature-analysis.public.local.json',[name for name,_value in probe.writes])
        probe.stack.close()
        budget_error=MemoryError('synthetic aggregate row budget exceeded')
        probe=NormalizationFixture(self,budget_error=budget_error)
        with self.assertRaises(MemoryError) as caught:probe.run()
        self.assertIs(caught.exception,budget_error)
        probe.load.assert_not_called();probe.analyzer.assert_not_called()
        self.assertIsNone(probe.closed()['meter']['feature_rows_completed'])


if __name__=='__main__':unittest.main()
