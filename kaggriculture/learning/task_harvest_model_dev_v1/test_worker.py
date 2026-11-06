"""Synthetic consumer/counter regressions; no engine, model, or file writes."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from learning.task_pair_dev_metrics_v1.test_consumer import (
    candidate, record, turn, unit_event, observations, audit)
from .worker import HarvestConsumer, Operations, PHASE_CAPS, _official_counts, _wrapper_code


def chosen(logits=(1.,2.), *, index=1):
    choices = [candidate(0,[1,0],op) for op in ('WATER','HARVEST')]
    result = record(op=choices[index]['op'],request=[choices[index]['op']],candidates=choices)
    result.update(task_model=dict(checkpoint_sha256='synthetic',logits=list(logits),
        argmax_index=index,chosen=choices[index]['id'],fallbacks=0),
        harvest_choice_extension=dict(added_candidate_ids=[choices[1]['id']],
            selected_virtual_harvest=index==1,source_water_id=choices[0]['id'] if index==1 else None))
    return result


def trace(step, records=()):
    value = turn(step,records)
    value.update(task_checkpoint_sha256='synthetic',
        model_counts=dict(controller_calls=step+1,task_forwards=1))
    return value


def label(status):
    value = unit_event(request=['HARVEST'],position=[1,0],status=status)
    value['before']['inventories'] = [[]]
    value['after']['inventories'] = [[('MELON',5)]] if status == 'executed' else [[]]
    value['tile_before'] = dict(kind='PLANT',crop='MELON',yield_units=5)
    value['tile_after'] = None if status == 'executed' else deepcopy(value['tile_before'])
    return value


class WorkerTests(unittest.TestCase):
    def test_virtual_choice_is_not_an_execution_until_official_label(self):
        for status,executed,units in (('executed',1,5),('no_effect',0,0)):
            with self.subTest(status=status):
                consumer = HarvestConsumer(0)
                for step in range(719):
                    consumer.trace_turn(trace(step,[chosen()] if step==0 else []))
                effect = label(status)
                consumer.sink(effect)
                value = consumer.finish(observations(100,100),observations(100,100),
                    ['DONE','DONE'],audit([effect]))['harvest_choice']
                self.assertEqual(value['counts']['selected_virtual_harvest'],1)
                self.assertEqual(value['selected_virtual_actual_executions'],executed)
                self.assertEqual(value['selected_virtual_harvested_melon_units'],units)
                self.assertEqual(value['selected_virtual_execution_status_counts'],{status:1})

    def test_missing_effect_rejects_instead_of_promoting_request(self):
        consumer = HarvestConsumer(0)
        for step in range(719):
            consumer.trace_turn(trace(step,[chosen()] if step==0 else []))
        with self.assertRaises(ValueError):
            consumer.finish(observations(100,100),observations(100,100),['DONE','DONE'],audit([]))

    def test_same_forward_argmax_tie_nonfinite_and_virtual_identity(self):
        valid = HarvestConsumer(0)
        valid.trace_turn(trace(0,[chosen((2.,2.),index=0)]))
        self.assertFalse(valid.virtual)
        invalid = [chosen((2.,2.),index=1),chosen((1.,float('nan')),index=1)]
        wrong = chosen();wrong['task_model']['chosen'] = wrong['teacher_label'];invalid.append(wrong)
        wrong = chosen();wrong['harvest_choice_extension']['source_water_id'] = 'missing';invalid.append(wrong)
        wrong = chosen();wrong['harvest_choice_extension']['added_candidate_ids'] = [wrong['teacher_label']];invalid.append(wrong)
        for value in invalid:
            with self.subTest(value=value):
                consumer = HarvestConsumer(0)
                with self.assertRaises(ValueError):consumer.trace_turn(trace(0,[value]))
                self.assertTrue(consumer.failed)

    def test_operation_phases_count_construction_and_nested_reset_separately(self):
        # These methods are no-op counters, not a simulated game/transition.
        class Environment:
            def __init__(self):self.reset()
            def reset(self):return 'reset'
            def step(self):return 'step'
        class Agent:
            def act(self,obs):return obs
        originals = Environment.__init__,Environment.reset,Environment.step,Agent.act
        counts={};operations=Operations(counts,lambda:True)
        with _official_counts(SimpleNamespace(Environment=Environment,Agent=Agent),
                SimpleNamespace(Agent=Agent),operations,Path(__file__).parent,dict(side=0)):
            live=Environment();live.reset()
            for _ in range(719):live.step()
            for who in ('actor_calls','rival_calls'):
                for _ in range(719):operations.call(who,lambda:None)
            with operations.at('reconstruction'):
                for who in ('actor_calls','rival_calls'):
                    for _ in range(719):operations.call(who,lambda:None)
                with operations.at('replay'):
                    replay=Environment()
                    for _ in range(719):replay.step()
                    with self.assertRaises(ValueError):replay.step()
            operations.complete()
        self.assertEqual(originals,(Environment.__init__,Environment.reset,Environment.step,Agent.act))
        self.assertEqual(sum(v['environment_constructions_returned'] for k,v in counts.items()
                             if 'environment_constructions' in PHASE_CAPS[k]),2)
        self.assertEqual(counts['live']['resets_returned']+counts['replay']['resets_returned'],3)

    def test_failed_operation_preserves_attempt_without_return_and_restores_phase(self):
        counts={};operations=Operations(counts,lambda:True)
        original=RuntimeError('synthetic')
        def fail():raise original
        with self.assertRaises(RuntimeError) as caught:
            with operations.at('reconstruction'):
                operations.call('rival_calls',fail)
        self.assertIs(caught.exception,original)
        self.assertEqual(operations.phase,'live')
        self.assertEqual(counts['reconstruction']['rival_calls_attempted'],1)
        self.assertEqual(counts['reconstruction']['rival_calls_returned'],0)

    def test_wrapper_uses_dual_model_policy_and_no_seed_literal(self):
        root=Path(__file__).resolve().parent/'synthetic-game'
        stage=root/'artifacts/synthetic-run'
        cfg=dict(crop_checkpoint='artifacts/models/crop.npz',crop_checkpoint_sha256='crop',game_limits={})
        plan=dict(game='synthetic-game',seed=987654321)
        code=_wrapper_code(root,stage,plan,cfg,root/'artifacts/models/task.npz','task')
        compile(code,'<synthetic wrapper>','exec')
        self.assertIn('BoundedHarvestAgent',code)
        self.assertNotIn(str(plan['seed']),code)
        self.assertIn('crop.npz',code)
        self.assertIn('task.npz',code)


if __name__ == '__main__':
    unittest.main()
