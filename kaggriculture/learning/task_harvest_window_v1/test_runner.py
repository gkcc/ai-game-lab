"""Coordinator counter and CLI boundaries; no private data or execution."""
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

from . import contract,runner


def documents():
    result=[]
    for source in contract.sources():
        calls={}
        for operation,value in (('arm',2),('actor_construct',2),('rival_construct',2),
                                ('restore',2),('official_step',96),('actor',576),('rival',576)):
            calls[operation+'_attempts']=calls[operation+'_returns']=value
        budget={'prefix':dict(controller_calls=480,task_forwards=480,task_rows=4800,crop_forwards=0,crop_rows=0),
                'suffix':dict(controller_calls=96,task_forwards=96,task_rows=960,crop_forwards=0,crop_rows=0)}
        result.append(dict(row=dict(source,status='paired'),target={'step':240},meter=dict(
            calls=calls,budget_by_scope=budget,unknown_budget_observations=0)))
    return result


class RunnerTests(unittest.TestCase):
    def test_complete_case_counts_include_prefix_and_factual_transitions(self):
        docs=documents()
        value=runner.check_complete_counts(docs,runner.aggregate_meters(docs),contract)
        self.assertEqual(value['official_transitions'],384)
        self.assertEqual(value['environment_constructions'],8)
        self.assertEqual(value['original_controller_calls'],2304)
        self.assertEqual(value['rival_calls'],2304)
        self.assertEqual(value['model_budget_totals']['task_forwards'],2304)

    def test_skipped_roles_have_no_fake_environment_or_controller_counts(self):
        docs=documents()
        for d in docs:
            d['row']['status']='no_eligible_target';d['target']=None
            d['meter']={'calls':{},'budget_by_scope':{'prefix':{},'suffix':{}},'unknown_budget_observations':0}
        value=runner.check_complete_counts(docs,runner.aggregate_meters(docs),contract)
        self.assertEqual(value['paired_roles'],0)
        self.assertEqual(value['official_transitions'],0)
        self.assertEqual(value['environment_constructions'],0)

    def test_missing_returns_extra_restores_unknown_costs_or_bool_counts_reject(self):
        original=documents()
        for operation in ('official_step','rival','actor','restore'):
            docs=deepcopy(original);docs[0]['meter']['calls'][operation+'_returns']-=1
            with self.assertRaises(ValueError):runner.check_complete_counts(docs,runner.aggregate_meters(docs),contract)
        docs=deepcopy(original);docs[0]['meter']['unknown_budget_observations']=1
        with self.assertRaises(ValueError):runner.check_complete_counts(docs,runner.aggregate_meters(docs),contract)
        docs=deepcopy(original);docs[0]['meter']['calls']['restore_attempts']=True
        with self.assertRaises(ValueError):runner.aggregate_meters(docs)

    def test_registration_binds_actual_protocol_limits_and_nontraining_scope(self):
        value=dict(schema=runner.SCHEMA,base_commit=runner.BASE,engine_commit=contract.ENGINE_COMMIT,
            engine_sha256=contract.ENGINE_SHA256,limits=contract.LIMITS,panel=contract.specification(),
            evaluation=dict(training=False,gpu=False,new_seeds=False,independent_evaluation=False,
                statistical_significance=False,promotion=False,terminal_profit=False,retries=False),
            binding_commitment_sha256='a'*64)
        runner.validate_registration(value,contract)
        changed=deepcopy(value);changed['panel']['horizon']=49
        with self.assertRaises(ValueError):runner.validate_registration(changed,contract)
        changed=deepcopy(value);changed['evaluation']['training']=True
        with self.assertRaises(ValueError):runner.validate_registration(changed,contract)

    def test_cleanup_failure_keeps_primary_and_never_turns_success_into_pass(self):
        def fail(_value):raise OSError('synthetic writer failure')
        with self.assertRaisesRegex(MemoryError,'original'):
            try:raise MemoryError('original')
            finally:
                import sys
                self.assertFalse(runner.close_preserving_primary(fail,{},sys.exc_info()[1]))
        with self.assertRaises(OSError):runner.close_preserving_primary(fail,{},None)

    def test_help_executes_nothing_and_bad_private_arguments_are_not_echoed(self):
        out,err=io.StringIO(),io.StringIO()
        with redirect_stdout(out),redirect_stderr(err),patch.object(runner,'run') as run:
            self.assertEqual(runner.main(['--help']),0);run.assert_not_called()
            self.assertEqual(runner.main(['--unknown','SYNTHETIC_PRIVATE_TOKEN']),1)
            self.assertEqual(runner.main(['--prepare','--case','F01']),1)
            run.side_effect=SystemExit(0)
            self.assertEqual(runner.main(['--registration-commit','a'*40]),1)
        self.assertNotIn('SYNTHETIC_PRIVATE_TOKEN',out.getvalue()+err.getvalue())


if __name__=='__main__':
    unittest.main()
