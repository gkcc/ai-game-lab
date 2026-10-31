"""Synthetic integration of a prepared gate with the real actor and CLI seam."""
from copy import deepcopy
from contextlib import nullcontext,redirect_stdout,redirect_stderr
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import io
import json
import sys
import unittest
from unittest.mock import patch

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_single_intervention_v1.test_intervention import inputs
from . import actor,gate,worker
from .test_actor import FakeBoundedAgent


def case():
    return dict(role='A05',slot='T01',opponent='market',side=0,window=0,rank=0,step=3,ordinal=1,
        input_sha256=ordered_hash(inputs(3)),original_choice='student',teacher_choice='teacher',
        seed=123,task_sha256='a'*64,crop_sha256='b'*64)


class DummyGame:
    game_limits={}
    def __enter__(self): return self
    def __exit__(self,*_args): return False


class WorkerTests(unittest.TestCase):
    def test_prepared_gate_is_accepted_by_actual_actor_for_both_arms(self):
        # In-memory filesystem: this does not access any private replay/model.
        actor_raw=Path(actor.__file__).read_bytes();gate_raw=Path(gate.__file__).read_bytes()
        stage=Path('synthetic');files={};events={}
        context=SimpleNamespace(_task_path='task',_crop_path='crop')
        def save(path,raw): files[str(path)]=raw
        def read_bytes(path): return files[str(path)]
        def read_text(path,**_kwargs):
            if path.name in events: return json.dumps(events[path.name])
            return files[str(path)].decode()
        def save_event(path,value):
            self.assertNotIn(path.name,events);events[path.name]=deepcopy(value)
        def source_bytes(path):
            if path==Path(actor.__file__): return actor_raw
            if path==Path(gate.__file__): return gate_raw
            return read_bytes(path)
        with patch.object(worker,'_source_context',return_value=context), \
             patch.object(worker,'_game',return_value=DummyGame()), \
             patch.object(worker,'_scan_target'),patch.object(worker,'_opponent',return_value=b'agent'), \
             patch.object(worker,'_relative_model',side_effect=lambda p:p), \
             patch.object(worker,'_write',side_effect=save),patch.object(Path,'mkdir'), \
             patch.object(Path,'resolve',lambda p,strict=False:p),patch.object(Path,'read_bytes',source_bytes), \
             patch.object(Path,'read_text',read_text),patch.object(actor,'_write',side_effect=save_event), \
             patch.object(actor,'_relative',side_effect=lambda root,value:root/value), \
             patch('learning.task_dagger_once_v1.policy.BoundedAgent',FakeBoundedAgent):
            details,names=worker._prepare(stage,case(),lambda:None)
            self.assertTrue(details['prepared'])
            self.assertIn('gate.local.json',names)
            config=json.loads(files[str(stage/'gate.local.json')])
            self.assertEqual(config['actor_source_sha256'],hashlib.sha256(actor_raw).hexdigest())
            self.assertNotIn('seed',config)
            for mode,expected in (('control',['student']*3),('intervene',['student','teacher','student'])):
                callback=actor.build(stage,'game',mode)
                for step in range(719):
                    result=callback({'step':step},{})
                    if step==3: self.assertEqual(result['choices'],expected)
                self.assertTrue(events[mode+'.complete.local.json']['target_verified'])

    def test_case_coordinate_and_identity_checks(self):
        good=case();self.assertIs(worker._case(good),good)
        for field,value in (('side',True),('rank',2),('ordinal',32),('seed',-1),
                            ('slot','T02'),('task_sha256','invalid')):
            bad=dict(good);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): worker._case(bad)

    def test_real_cli_adapter_removes_seed_before_agent_execution(self):
        probe=case();stage=Path('synthetic'); seen=[]
        fake=SimpleNamespace(ROOT=worker.GAME, argparse=argparse, native_output=None, fail=False)
        def main():
            parser=fake.argparse.ArgumentParser()
            for name in ('bot','opponent','side','seed','name','output-root'):
                parser.add_argument('--'+name)
            parsed=parser.parse_args()
            seen.append((parsed.seed,list(sys.argv)))
            print('sensitive path and seed must not escape')
            if fake.fail: raise RuntimeError('private seed or path')
            return 0
        fake.main=main
        result=dict(passed=True,done=True,statuses=['DONE','DONE'],recorded_states=720,action_turns=719,
            agent_errors=[],seed=probe['seed'],bot_side=probe['side'],
            engine_sha256=worker.screen_experiment.ENGINE_SHA,engine_matches_official_commit=True,
            upstream_commit=worker.screen_experiment.ENGINE_COMMIT)
        complete=dict(passed=True,action_turns=719,target_verified=True)
        original_argv=list(sys.argv)
        spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _mod:None))
        def children(path,pattern): return [path/'replay.json',path/'result.json']
        with patch.object(worker,'_verify_bundle'),patch.object(worker.os,'chdir'), \
             patch.object(worker.importlib.util,'spec_from_file_location',return_value=spec), \
             patch.object(worker.importlib.util,'module_from_spec',return_value=fake), \
             patch.object(worker,'_json',side_effect=lambda p:result if p.name=='result.json' else complete), \
             patch.object(Path,'exists',return_value=False),patch.object(Path,'rglob',children), \
             patch.object(Path,'is_file',return_value=True), \
             patch.object(worker.gzip,'open',return_value=nullcontext(object())), \
             patch.object(worker.samples,'scan',return_value={'lines':719}), \
             patch.object(worker.trace_budget,'validate_trace',return_value={}):
            output,errors=io.StringIO(),io.StringIO()
            with redirect_stdout(output),redirect_stderr(errors):
                details,names=worker._run_match(stage,probe,'control',lambda:None)
                fake.fail=True
                with self.assertRaises(RuntimeError): worker._run_match(stage,probe,'control',lambda:None)
            self.assertEqual((output.getvalue(),errors.getvalue()),('',''))
        self.assertEqual(seen,[(str(probe['seed']),[str(worker.GAME/'run_match.py')])]*2)
        self.assertEqual(list(sys.argv),original_argv)
        self.assertTrue(details['both_done'])
        self.assertIn('games/control/replay.json',names)

    def test_budget_checker_rejects_33_callbacks_scanner_accepts(self):
        def encoded(first_count):
            values={key:0 for key in gate.BUDGET_KEYS}
            rows=[]
            for step in range(719):
                turn={key:0 for key in values};turn['controller_calls']=1
                count=first_count if step==0 else 0
                turn['task_forwards']=turn['task_rows']=count
                for key in values: values[key]+=turn[key]
                record=dict(source='callback',inputs=dict(candidates=[dict(id='a')]),
                            teacher_label='a',chosen='a')
                rows.append(dict(schema_version='task-choice-v1',step=step,
                    task_checkpoint_sha256='a'*64,checkpoint_sha256='b'*64,
                    load_errors=0,inference_errors=0,fallbacks=0,records=[record]*count,
                    crop_records=[],model_counts=dict(values),turn_model_counts=turn))
            return (''.join(json.dumps(row)+'\n' for row in rows)).encode()
        raw=encoded(33)
        self.assertEqual(worker.samples.scan(io.BytesIO(raw),task_sha256='a'*64,crop_sha256='b'*64)['lines'],719)
        original=worker.trace_budget.SAMPLE_TURNS
        valid_case=dict(task_sha256='a'*64,crop_sha256='b'*64)
        with self.assertRaises(ValueError):
            worker._validate_trace_budget(io.BytesIO(raw),valid_case,lambda:None)
        self.assertIs(worker.trace_budget.SAMPLE_TURNS,original)
        self.assertIsNone(worker._validate_trace_budget(io.BytesIO(encoded(32)),valid_case,lambda:None))
        self.assertIs(worker.trace_budget.SAMPLE_TURNS,original)

    def test_phase_rechecks_runtime_source_identity_after_execution(self):
        stage=Path('virtual')/'artifacts'/'case'
        with patch.object(worker,'GAME',Path('virtual')), \
             patch.object(Path,'resolve',lambda p,strict=False:p),patch.object(Path,'is_dir',return_value=True), \
             patch.object(worker,'_engine'),patch.object(worker,'_verify_bundle'), \
             patch.object(worker,'_source_context',side_effect=[object(),ValueError('runtime changed')]), \
             patch.object(worker,'_audit',return_value=({'action_turns':719,'shed_order_frames_checked':720},())):
            with self.assertRaises(ValueError):
                worker.run_phase(stage=stage,phase='control_audit',case=case(),guard=lambda:None)


if __name__=='__main__': unittest.main()
