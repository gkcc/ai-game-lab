"""Two-opponent routing and immutable-teacher verifier integration regressions."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import contract, runner
from learning.task_teacher_market_dev_v1 import entry as teacher
from learning.task_teacher_market_dev_v1.test_entry import fixture


def accept(plan, job, bundle, hashes):
    return teacher.verify_bundle(plan, job, bundle, True, crop_sha='c'*64, engine_sha='d'*64,
        storage_limit=100, native_scope='NUL scope', receipt_hashes=hashes,
        verify_files=lambda _p,_r:True)


class RunnerTests(unittest.TestCase):
    def test_job_adapter_keeps_the_actual_threshold_read_without_changing_it(self):
        values = iter((7, 6, 3, 8))
        class API:
            def available(self): return next(values)
        observations = dict(count=0,last=None,minimum=None,first_below_floor=None,sample_error=None)
        times = iter((1., 2., 3., 4.))
        api = runner.observed_api_type(API,4,observations,clock=lambda:next(times))()
        self.assertEqual([api.available() for _ in range(4)],[7,6,3,8])
        self.assertEqual(observations['count'],4)
        self.assertEqual(observations['first_below_floor'],dict(available_bytes=3,sample_monotonic=3.))
        self.assertEqual(observations['minimum'],observations['first_below_floor'])
        self.assertEqual(observations['last'],dict(available_bytes=8,sample_monotonic=4.))
        self.assertIsNone(observations['sample_error'])

    def test_failed_job_memory_read_remains_unknown_and_propagates(self):
        class API:
            def available(self): raise OSError('synthetic read failure')
        observations = dict(count=0,last=None,minimum=None,first_below_floor=None,sample_error=None)
        api = runner.observed_api_type(API,4,observations,clock=lambda:11.)()
        with self.assertRaises(OSError): api.available()
        self.assertEqual(observations['count'],0)
        self.assertIsNone(observations['last'])
        self.assertEqual(observations['sample_error'],dict(error_kind='OSError',sample_monotonic=11.))

    def test_closure_serialization_preserves_stop_sample_before_later_recovery(self):
        latest = dict(available_bytes=3*1024**3, sample_monotonic=17.)
        error = TimeoutError('synthetic gate timeout')
        error.last_available_bytes = latest['available_bytes']
        def recovered():
            latest.update(available_bytes=7*1024**3, sample_monotonic=31.)
        value = json.loads(json.dumps(runner.closing_memory(error, latest, recovered)))
        boundary = value['failure_boundary_observation']
        self.assertEqual(boundary['last_observation_before_close'],
                         dict(available_bytes=3*1024**3, sample_monotonic=17.))
        self.assertEqual(boundary['gate_last_available_bytes'], 3*1024**3)
        self.assertEqual(value['memory_at_close']['available_bytes'], 7*1024**3)
        self.assertEqual(value['memory_at_close']['sample_monotonic'], 31.)

    def test_teacher_verifier_accepts_round2_without_admitting_model_task_calls(self):
        plan, job, bundle, hashes = fixture()
        name = plan['game'].replace('-market-', '-round2-')
        plan.update(game=name, opponent='round2')
        job.update(game=name, opponent='round2')
        for value in bundle.values(): value['game'] = name
        bundle['row']['opponent'] = 'round2'
        with contract.teacher_adapter(teacher):
            self.assertEqual(accept(plan, job, bundle, hashes), bundle['row'])
            for key, bad in (('task_forwards',1),('task_rows',1),('crop_forwards',0),('controller_calls',718)):
                changed = deepcopy(bundle)
                for which in ('primary_model_counts','live_model_counts'):
                    changed['audit'][which][key] = bad
                with self.subTest(key=key), self.assertRaises(ValueError):
                    accept(plan, job, changed, hashes)

    def test_child_routing_uses_selected_opponent_and_zero_task_identity(self):
        stage = Path.cwd().resolve() / 'synthetic-uncreated-stage'
        for opponent in contract.OPPONENTS:
            plans = contract.plans(SimpleNamespace(seeds=(12,34)))
            plan = next(p for p in plans if p['opponent']==opponent)
            chosen = 'a'*64 if opponent=='market' else 'b'*64
            written, hashed, calls = {}, [], []
            def hash_file(root,path,limit):
                hashed.append(path)
                return chosen if path.parent.name=='sources' else 'c'*64
            dm = SimpleNamespace(MAX_SOURCE=1000, _read=lambda *_:b'worker', _hash=hash_file,
                _write_new=lambda root,path,value,**kw:written.update({path:dict(value)}))
            worker = SimpleNamespace(WORKER_RELATIVE='worker.py',MAX_WORKER_BYTES=1000,MAX_JOB_BYTES=1000,
                _single_python_thread=lambda:None,_quiet_output=nullcontext)
            module = SimpleNamespace(wrapper=lambda *_:stage/'agents'/(plan['game']+'.py'))
            expected = [None]
            def run_job(out,key):
                self.assertEqual(expected[0]['rival_sha256'],chosen)
                self.assertIsNone(expected[0]['task_checkpoint_sha256'])
                calls.append(key)
            module.job = run_job
            @contextmanager
            def bound(*args):
                self.assertEqual(fake.validate_plan(plan,plan),plan)
                yield module,{},expected
            fake = SimpleNamespace(validate_plan=lambda *_:None,bound_worker=bound)
            manifest = SimpleNamespace(panel=SimpleNamespace(worker_sha=runner.sha(b'worker')),
                opponent=lambda role:('bots/'+role+'.py','a'*64 if role=='market' else 'b'*64))
            with patch.object(runner,'hidden_official_entry',lambda *_:nullcontext()):
                runner.execute_child(dict(driver=dm,worker=worker,teacher=fake),manifest,stage,plan,
                    lambda:True,contract)
            self.assertEqual(calls,[plan['game']])
            self.assertIn(stage/'sources'/(opponent+'.py'),hashed)
            self.assertNotIn(stage/'sources'/('round2.py' if opponent=='market' else 'market.py'),hashed)
            self.assertEqual(len(written),1)

    def test_registration_rejects_modified_scope_and_resource_contract(self):
        reg = dict(schema=runner.SCHEMA,base_commit=runner.BASE,limits=deepcopy(contract.LIMITS),
            engine_commit='f994f2050d58d84f649329c1e35ce09ada7007c5',
            engine_sha256='bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e',
            panel=dict(cohorts=contract.COHORTS,opponents=list(contract.OPPONENTS),sides=list(contract.SIDES),
                policy='teacher_task',games=16,official_replays=16,source_student_games=32,
                seed_clusters=4,scope='consumed_development'),
            evaluation=dict(training=False,gpu=False,new_seeds=False,independent_evaluation=False,
                statistical_significance=False,promotion=False,transaction_cash_reconciliation=False,retries=False),
            binding_commitment_sha256='a'*64,old_result_commit='0e9734a0109a0ee4822ef1310e8b56fac21eaa48')
        runner.validate_registration(reg,contract)
        for section,key,value in (('limits','max_games',17),('panel','opponents',['market']),
                                   ('evaluation','independent_evaluation',True),
                                   ('limits','max_attempts',True)):
            bad=deepcopy(reg);bad[section][key]=value
            with self.subTest(section=section,key=key),self.assertRaises(ValueError):
                runner.validate_registration(bad,contract)


if __name__ == '__main__':
    unittest.main()
