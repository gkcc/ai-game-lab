"""Synthetic F adapters and actual one-shot actor contract; no real models."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_single_intervention_v1.test_intervention import inputs
from learning.task_teacher_counterfactual_pair_v1 import actor
from learning.task_teacher_counterfactual_pair_v1.test_actor import FakeBoundedAgent
from . import sources, worker
from .test_sources import context_fixture


def case_for(context):
    plan = context.private_source_plans()[0]
    return dict(role=plan.role,slot=plan.slot,opponent=plan.opponent,side=plan.side,
        cohort=plan.cohort,alias=plan.alias,source_game=plan.source_game,
        source_identity_sha256=plan.identity_sha256,seed=plan.seed,task_sha256=plan.task_sha256,
        crop_sha256=plan.crop_sha256,window=0,rank=0,step=3,ordinal=1,
        input_sha256=ordered_hash(inputs(3)),original_choice='student',teacher_choice='teacher')


class WorkerTests(unittest.TestCase):
    def test_candidate_is_bound_to_full_f_source_and_rejects_old_roles_or_changed_scope(self):
        context = context_fixture(); case = case_for(context)
        self.assertEqual(worker.validate_case(case,context),case)
        for key,value in (('role','A05'),('slot','T02'),('cohort','C02'),('alias','S03'),
                          ('source_game',case['source_game'].replace('baseline','control')),
                          ('source_identity_sha256','0'*64),('side',True),('seed',999),
                          ('task_sha256','0'*64),('step',120),('ordinal',32)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                worker.validate_case(dict(case,**{key:value}),context)

    def test_source_adapters_restore_all_legacy_functions_after_failure(self):
        context=context_fixture();case=case_for(context)
        names=('_case','_source_context','_game','_scan_target','_opponent','_verify_bundle')
        original={name:getattr(worker.legacy,name) for name in names}
        fake=SimpleNamespace(plan=context.private_source_plans()[0])
        with patch.object(sources,'target_record',side_effect=ValueError('synthetic failed target')):
            with self.assertRaises(ValueError):
                with worker.adapted_worker(context) as adapted:
                    self.assertEqual(adapted._case(case),case)
                    adapted._scan_target(fake,case,lambda:None)
        for name in names: self.assertIs(getattr(worker.legacy,name),original[name])
        with worker.adapted_worker(context):
            with self.assertRaises(ValueError):
                with worker.adapted_worker(context): pass
        for name in names: self.assertIs(getattr(worker.legacy,name),original[name])

    def test_f_preparation_feeds_seed_free_gate_to_real_actor_with_synthetic_network(self):
        context=context_fixture();case=case_for(context);plan=context.private_source_plans()[0]
        legacy=worker.legacy;stage=Path('synthetic-uncreated-f-stage');files={};events={}
        actor_raw=Path(actor.__file__).read_bytes();gate_raw=Path(legacy.gate.__file__).read_bytes()
        fake=SimpleNamespace(plan=plan,game_limits=sources.verify_limits(),opponent_bytes=lambda _g:b'opponent')
        def source_bytes(path):
            if path==Path(actor.__file__): return actor_raw
            if path==Path(legacy.gate.__file__): return gate_raw
            return files[str(path)]
        def read_text(path,**_kwargs):
            return json.dumps(events[path.name]) if path.name in events else files[str(path)].decode()
        def save_event(path,value):
            self.assertNotIn(path.name,events);events[path.name]=deepcopy(value)
        with patch.object(context,'verify_assets',return_value=True), \
             patch.object(sources,'read_game',return_value=nullcontext(fake)), \
             patch.object(sources,'target_record',return_value={}), \
             patch.object(legacy,'_relative_model',side_effect=lambda p:'task' if p==context._task_path else 'crop'), \
             patch.object(legacy,'_write',side_effect=lambda p,b:files.update({str(p):b})), \
             patch.object(Path,'mkdir'),patch.object(Path,'resolve',lambda p,strict=False:p), \
             patch.object(Path,'read_bytes',source_bytes),patch.object(Path,'read_text',read_text), \
             patch.object(actor,'_relative',side_effect=lambda root,value:root/value), \
             patch.object(actor,'_write',side_effect=save_event), \
             patch('learning.task_dagger_once_v1.policy.BoundedAgent',FakeBoundedAgent):
            with worker.adapted_worker(context) as adapted:
                details,names=adapted._prepare(stage,case,lambda:None)
            config=json.loads(files[str(stage/'gate.local.json')])
            self.assertFalse(set(config)&{'seed','cohort','alias','source_game','role','side','opponent'})
            self.assertEqual(config['actor_source_sha256'],hashlib.sha256(actor_raw).hexdigest())
            for mode,expected in (('control',['student']*3),('intervene',['student','teacher','student'])):
                callback=actor.build(stage,'synthetic-game',mode)
                for step in range(719):
                    result=callback({'step':step},{})
                    if step==3: self.assertEqual(result['choices'],expected)
                self.assertTrue(events[mode+'.complete.local.json']['target_verified'])
        self.assertTrue(details['prepared']);self.assertIn('gate.local.json',names)

    def test_target_input_must_match_legally_observed_state_and_configuration(self):
        context=context_fixture();case=case_for(context);plan=context.private_source_plans()[0]
        value=inputs(3)
        value['configuration']={}
        case['input_sha256']=ordered_hash(value)
        record=dict(source='callback',chosen='student',teacher_label='teacher',inputs=value)
        candidate=dict(step=3,ordinal=1,third_candidate=None,
            document=dict(step=3,records=[deepcopy(record),record]),previous_document=None)
        windows=[[candidate],[],[],[],[],[]]
        fake=SimpleNamespace(plan=plan,replay=dict(steps=[None]*720,configuration={},info={'seed':plan.seed}),
            scan_candidates=lambda **_kw:({'lines':719,'absolute_budget_checked':True},windows))
        view=SimpleNamespace(observation=deepcopy(value['observation']),configuration=deepcopy(value['configuration']))
        with patch.object(sources,'observation',return_value={}),patch.object(sources,'observe',return_value=view):
            self.assertIs(sources.target_record(fake,case,lambda:None),record)
            record['inputs']['observation']['opponent_private']={'hidden':1}
            # Update the case digest to show that matching a self-provided hash
            # cannot make an unobservable input acceptable.
            changed=dict(case,input_sha256=ordered_hash(record['inputs']))
            with self.assertRaises(ValueError): sources.target_record(fake,changed,lambda:None)

    def test_bundle_rejects_replaced_opponent_or_retargeted_model(self):
        context=context_fixture();case=case_for(context);stage=Path('synthetic-uncreated-stage')
        config=dict(limits=sources.verify_limits(),task_checkpoint='task',crop_checkpoint='crop',
                    actor_source_sha256='1'*64,gate_source_sha256='2'*64)
        def digest(path,*_args):
            if path.name=='actor.py': return '1'*64
            if path.name=='gate.py': return '2'*64
            return 'c'*64
        with patch.object(worker.legacy,'_relative_model',side_effect=lambda p:'task' if p==context._task_path else 'crop'), \
             patch.object(worker.legacy,'_sha_file',side_effect=digest):
            self.assertEqual(worker.verify_bundle(stage,case,lambda:None,context,lambda *_:config),config)
            changed=dict(config,task_checkpoint='other-task')
            with self.assertRaises(ValueError):
                worker.verify_bundle(stage,case,lambda:None,context,lambda *_:changed)
        with patch.object(worker.legacy,'_relative_model',side_effect=lambda p:'task' if p==context._task_path else 'crop'), \
             patch.object(worker.legacy,'_sha_file',return_value='0'*64):
            with self.assertRaises(ValueError):
                worker.verify_bundle(stage,case,lambda:None,context,lambda *_:config)


if __name__ == '__main__':
    unittest.main()
