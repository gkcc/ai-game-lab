"""Pure orchestration/measurement contracts; no private source or model reads."""
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_pair_dev_games_v1.panel import Panel
from learning.task_pair_dev_games_v1.test_panel import synthetic_inputs
from . import contract,runner
from .decoder import choose


def encoded(value): return (json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode()


def candidates():
    return [dict(id=f'{u}:{x}:0:WATER',unit=u,target=[x,0],op='WATER') for u,x in ((0,0),(1,0),(1,1))]


def trace():
    graph=candidates();selected=choose(graph,[8.,10.,9.]);turns=[]
    for step in range(719):
        records=([dict(inputs={'candidates':graph},source='callback',chosen=selected.choice_id)] if step==0 else [])
        decoded=([dict(ordinal=0,original_network_choice=graph[1]['id'],decoder_choice=selected.choice_id,
            matching_ids=list(selected.matching_ids),matching_cardinality=2,matching_logit_sum=17.,logits=[8.,10.,9.])]
            if step==0 else [])
        counts=dict(task_forwards=1,task_rows=3,crop_forwards=step+1,crop_rows=step+1,
            gate_evaluations=0,controller_calls=step+1,teacher_proposals=0)
        local=dict(counts,task_forwards=int(step==0),task_rows=3 if step==0 else 0,
                   crop_forwards=1,crop_rows=1,controller_calls=1)
        turns.append(dict(step=step,decoder_schema_version='task-joint-decoder-v1',task_checkpoint_sha256='a'*64,
            load_errors=0,inference_errors=0,fallbacks=0,records=records,decoder_records=decoded,
            model_counts=counts,turn_model_counts=local))
    return turns


def replay():
    steps=[]
    for step in range(720):
        frame=[]
        for player in (0,1):
            frame.append(dict(observation=dict(step=step,player=player,
                farms=[{'money':100},{'money':100}],market={'price':5},remainingOverageTime=60,
                private=dict(shed={'WHEAT':1,'CORN':1},inventories=[{'WHEAT':1,'CORN':1}])),
                reward=100 if step==719 else 0,status='DONE' if step==719 else 'ACTIVE',
                action=dict(farmer=['PASS'],hands=[],market=[])))
        steps.append(frame)
    return dict(steps=steps,configuration={'seed':None,'episodeSteps':720},info={'seed':17})


def case_rows():
    return [dict(role=role,opponent=opponent,side=side,
        baseline=dict(own=20,opponent=10,margin=10,score=1),
        candidate=dict(own=20,opponent=10,margin=10,score=1),
        decoder_changed_callbacks=0,first_request_difference=None,first_state_difference=None)
        for role,_alias,opponent,side in contract.CASES]


class RunnerTests(unittest.TestCase):
    def test_single_attempt_rejects_any_old_stage_or_owner_marker(self):
        runner.untouched(exists=lambda _path:False)
        for blocked in (runner.STAGE,runner.PRIVATE/'attempt.local.json',runner.PRIVATE/'owner.local.json',
                        runner.PRIVATE/'closed.local.json'):
            with self.subTest(kind=blocked.name),self.assertRaises(ValueError):
                runner.untouched(exists=lambda path:path==blocked)

    def test_real_panel_binding_requires_all_four_exact_plans(self):
        base_panel=Panel.from_private_contract(**synthetic_inputs());panel=contract.JointPanel(base_panel)
        context=SimpleNamespace(manifests={'C01':SimpleNamespace(panel=base_panel)},identities={'synthetic':True})
        infra=SimpleNamespace(read=lambda _path:b'default')
        modules=dict(sources=SimpleNamespace(load=lambda **_kw:context),driver=object())
        base=SimpleNamespace(validate_assets=lambda *_args:True)
        binding=dict(completed_sources=context.identities,default_sha256=runner.sha(b'default'),micro_files={},plans=list(panel.plans()))
        with patch.object(runner,'micro_evidence',return_value={}):
            actual,adapted=runner.source_context(infra,{'public_sources':{}},modules,base,contract,binding,lambda:True)
            self.assertIs(actual,context);self.assertEqual(adapted.task_model(panel.plans()[0]),base_panel.models[0])
            for wrong in (binding['plans'][:-1],binding['plans']+[binding['plans'][0]],list(reversed(binding['plans']))):
                with self.assertRaises(ValueError):
                    runner.source_context(infra,{'public_sources':{}},modules,base,contract,dict(binding,plans=wrong),lambda:True)

    def test_complete_trace_counts_real_choices_and_actual_original_forwards(self):
        turns=trace();summary=runner.trace_summary(iter(turns),'a'*64,turns[-1]['model_counts'])
        self.assertEqual(summary['trace_turns'],719)
        self.assertEqual(summary['decoder_callbacks'],1)
        self.assertEqual(summary['decoder_changed_callbacks'],1)
        self.assertEqual(summary['decoder_candidate_rows'],3)
        self.assertEqual(summary['original_network_forward_calls'],1)
        self.assertEqual(summary['additional_network_forwards'],0)

    def test_missing_extra_mismatched_or_bool_trace_counts_are_rejected(self):
        original=trace();primary=original[-1]['model_counts']
        variants=[original[:-1],original+[original[-1]]]
        changed=deepcopy(original);changed[0]['decoder_records'][0]['decoder_choice']=candidates()[0]['id'];variants.append(changed)
        changed=deepcopy(original);changed[0]['decoder_records'][0]['original_network_choice']=candidates()[0]['id'];variants.append(changed)
        changed=deepcopy(original);changed[0]['model_counts']['task_forwards']=True;variants.append(changed)
        changed=deepcopy(original);changed[0]['decoder_records']=[];variants.append(changed)
        for turns in variants:
            with self.assertRaises(ValueError):runner.trace_summary(iter(turns),'a'*64,primary)

    def test_physical_comparison_excludes_actions_logs_and_timing_but_keeps_market(self):
        first=replay();before=runner.compact_replay(first,lambda:True)
        second=deepcopy(first);second['steps'][1][0]['action']['farmer']=['MOVE','EAST']
        second['steps'][1][0]['logs']='synthetic unrelated output'
        second['steps'][1][0]['observation']['remainingOverageTime']=2
        after=runner.compact_replay(second,lambda:True)
        self.assertEqual(runner.compare_compacts(before,after,0),
                         dict(first_request_difference=0,first_state_difference=None))
        second['steps'][1][0]['observation']['market']['price']=6
        after=runner.compact_replay(second,lambda:True)
        self.assertEqual(runner.compare_compacts(before,after,0)['first_state_difference'],1)

    def test_shed_and_backpack_order_remain_material(self):
        source=replay();before=runner.compact_replay(source,lambda:True)
        for kind in ('shed','inventory'):
            changed=deepcopy(source);changed['steps'][1][0]['action']['farmer']=['MOVE','EAST']
            private=changed['steps'][1][0]['observation']['private']
            if kind=='shed':private['shed']={'CORN':1,'WHEAT':1}
            else:private['inventories']=[{'CORN':1,'WHEAT':1}]
            after=runner.compact_replay(changed,lambda:True)
            self.assertEqual(runner.compare_compacts(before,after,0)['first_state_difference'],1)

    def test_rival_cannot_differ_before_own_request_or_state_before_request(self):
        first=runner.compact_replay(replay(),lambda:True)
        second=deepcopy(first);second['requests'][0]=('same',second['requests'][0][1]);second['states'][0]=('wrong','wrong')
        with self.assertRaises(ValueError):runner.compare_compacts(first,second,0)
        second=deepcopy(first);second['requests'][0]=(second['requests'][0][0],'wrong')
        with self.assertRaises(ValueError):runner.compare_compacts(first,second,0)

    def test_four_case_projection_rejects_omission_duplicate_and_reordering(self):
        rows=case_rows();self.assertEqual(contract.project(rows)['cases'],4)
        for changed in (rows[:-1],rows+[rows[0]],[rows[0],rows[0],*rows[2:]],list(reversed(rows))):
            with self.assertRaises(ValueError):contract.project(changed)

    def test_close_write_failure_preserves_primary_and_cannot_turn_success_into_pass(self):
        def failed(_value):raise OSError('synthetic close write failure')
        with self.assertRaisesRegex(MemoryError,'primary'):
            try:raise MemoryError('primary')
            finally:
                import sys
                self.assertFalse(runner.close_preserving_primary(failed,{},sys.exc_info()[1]))
        with self.assertRaises(OSError):runner.close_preserving_primary(failed,{},None)
        saved=[];self.assertTrue(runner.close_preserving_primary(saved.append,{'closed':True},None))
        self.assertEqual(saved,[{'closed':True}])

    def test_seal_requires_closed_complete_four_case_evidence_and_byte_hashes(self):
        commit,commitment='a'*40,'b'*64;rows=case_rows();inputs={'accepted':[]}
        values={'ledger':dict(registration_commit=commit,binding_sha256=commitment,rows=rows),
            'projection':contract.project(rows),
            'results-attempt':dict(registration_commit=commit,binding_sha256=commitment,retry_allowed=False),
            'results-closed':dict(schema='task-joint-decoder-results-closed-v1',registration_commit=commit,
                binding_sha256=commitment,outcome='completed',error_kind=None,completed_comparisons=4,retry_allowed=False)}
        files={name:encoded(value) for name,value in values.items()}
        seal=dict(schema='task-joint-decoder-certificate-v1',registration_commit=commit,binding_sha256=commitment,
            technical_passed=True,games=4,official_replays=4,game_inputs=inputs,
            files={name:runner.sha(raw) for name,raw in files.items()})
        files['certificate']=encoded(seal)
        infra=SimpleNamespace(read=lambda path:files[path.name.removesuffix('.local.json')],decode=json.loads)
        self.assertTrue(runner.verify_seal(infra,contract,commit,commitment,inputs)['technical_acceptance'])
        files['ledger']+=b' '
        with self.assertRaises(ValueError):runner.verify_seal(infra,contract,commit,commitment,inputs)
        files['ledger']=encoded(values['ledger']);values['results-closed']['completed_comparisons']=3
        files['results-closed']=encoded(values['results-closed'])
        seal['files']['results-closed']=runner.sha(files['results-closed']);files['certificate']=encoded(seal)
        with self.assertRaises(ValueError):runner.verify_seal(infra,contract,commit,commitment,inputs)

    def test_help_is_success_but_execution_systemexit_and_private_argument_errors_are_not(self):
        out,err=io.StringIO(),io.StringIO()
        with redirect_stdout(out),redirect_stderr(err),patch.object(runner,'run') as run:
            self.assertEqual(runner.main(['--help']),0);run.assert_not_called()
            self.assertEqual(runner.main(['--unknown','SYNTHETIC_PRIVATE_VALUE']),1)
            run.side_effect=SystemExit(0)
            self.assertEqual(runner.main(['--registration-commit','a'*40]),1)
        self.assertNotIn('SYNTHETIC_PRIVATE_VALUE',out.getvalue()+err.getvalue())


if __name__=='__main__':
    unittest.main()
