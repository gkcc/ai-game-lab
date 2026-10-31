"""Synthetic entry tests only: no real source, model, Job, training or game."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from learning.task_state_residual_v1 import test_experiment as old_test
from . import analysis, experiment as e


def guard():pass


def prior_result():
    value=old_test.fake_result(e.STATE_COMMIT)
    coverage=value['coverage']; split=coverage['split']
    coverage['eligible_by_pool']['new_train']=42
    coverage['new_roles'].update(A03=21,A04=21)
    coverage['new_role_windows'].update(A03=6,A04=6)
    split['retained_by_pool']['new_train']=42
    split['duplicate_removed_by_pool']['new_train']=0
    split['retained_count']+=26
    split['train_component_count']+=26
    split['all_feature_component_count']+=26
    return value


def metrics():
    rows=[]
    for role in ('A03','A04'):
        rows += [dict(role=role,target_op='WATER',baseline_correct=False,candidate_correct=False,
                      changed=False,op_flips=False,base_nll=1.,candidate_nll=1.,base_margin=-1.,
                      candidate_margin=-1.,margin_increased=False,margin_decreased=False,margin_within_tolerance=True)
                 for _ in range(21)]
    value=dict(overall=analysis._bucket(rows),
        by_role={r:analysis._bucket([x for x in rows if x['role']==r]) for r in analysis.ROLES},
        by_target_operation={op:analysis._bucket([x for x in rows if x['target_op']==op]) for op in analysis.OPS})
    analysis.validate(value)
    return value


def fake_result(head='a'*40):
    return dict(schema='state-train-fit-result-v1',registration_commit=head,source_groups=42,
        fit_input_membership=True,student_mismatches=0,metrics=metrics(),baseline_sha256='b'*64,
        params_sha256='c'*64,train_sha256='d'*64,device='cpu',training=False,model_updates=0,
        new_games=0,gpu=False,independent_evaluation=False,promotion=False,
        software=dict(python='3.12',numpy='1.0'))


class EntryTests(unittest.TestCase):
    def test_registration_freezes_margin_classification_tolerance(self):
        document=e._registration({},'a'*64)
        self.assertEqual(document['margin_delta_abs_tolerance'],1e-6)
        self.assertEqual(document['margin_classification'],
                         'above_tolerance_below_negative_tolerance_otherwise_within_tolerance')
        self.assertEqual(analysis.MARGIN_TOLERANCE,document['margin_delta_abs_tolerance'])

    def test_42_membership_and_original_item_identity(self):
        prior=prior_result()
        self.assertTrue(e._all_train_in_fit(prior))
        for mutation in ('removed','filtered','mismatch','source'):
            bad=deepcopy(prior)
            if mutation=='removed':bad['coverage']['split']['retained_by_pool']['new_train']-=1
            if mutation=='filtered':bad['coverage']['filtered']['new_train']['student_same']=1
            if mutation=='mismatch':bad['train_student_checked']['mismatches']=1
            if mutation=='source':bad['registration_commit']='f'*40
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):e._all_train_in_fit(bad)
        # Existing frozen producer keeps or removes original objects; it does
        # not synthesize replacement groups. This supports the count proof.
        examples=old_test.all_examples()
        pools,_=e.state._filter_pools(examples,np.zeros(373,np.float64),np.ones(373,np.float64),guard)
        originals=[x for x in examples if x.item.pool=='new_train']
        self.assertEqual(len(originals),42)
        self.assertEqual([id(x) for x in pools['new_train']],[id(x) for x in originals])

    def test_training_archive_digest_checked_before_decode(self):
        raw=b'synthetic training archive'; arrays={'teacher_target':[0]*42}
        binding={'train_sha256':e._sha(raw)}
        with patch.object(e,'_read',return_value=raw), \
             patch.object(e.source,'_load_split_bytes',return_value=arrays) as decode:
            self.assertIs(e._train_arrays(binding,guard),arrays)
            decode.assert_called_once_with(raw,'train',guard)
            decode.reset_mock();binding['train_sha256']='a'*64
            with self.assertRaises(ValueError):e._train_arrays(binding,guard)
            decode.assert_not_called()

    def test_train_score_checks_student_and_only_model_inputs(self):
        arrays=old_test.source_arrays('train')
        network=SimpleNamespace(score=Mock(return_value=np.array([0.,1.],np.float64)))
        groups,students,roles=e._score_train(arrays,network,guard)
        self.assertEqual((len(groups),len(students),len(roles)),(42,42,42))
        self.assertEqual(network.score.call_count,42)
        self.assertTrue(all(len(call.args)==2 for call in network.score.call_args_list))
        network.score=Mock(return_value=np.array([1.,0.],np.float64))
        with self.assertRaises(ValueError):e._score_train(arrays,network,guard)

    def test_result_whitelist_and_sanitized_projection(self):
        result=fake_result()
        self.assertEqual(e._result(e.previous._encoded(result),'a'*40),result)
        public=e.public_projection(result)
        serialized=json.dumps(public)
        for forbidden in ('baseline_sha256','params_sha256','train_sha256','software'):
            self.assertNotIn(forbidden,serialized)
        bad=deepcopy(result);bad['metrics']['overall']['baseline_correct']=1
        with self.assertRaises(ValueError):e._result(e.previous._encoded(bad),'a'*40)
        bad=deepcopy(result);bad['metrics']['unexpected']='private'
        with self.assertRaises(ValueError):e._result(e.previous._encoded(bad),'a'*40)


class ChildTests(unittest.TestCase):
    def exercise(self, mismatch=False):
        from learning.task_teacher_dev_deviation_v1 import membership
        from learning.task_pair_fresh_multi_v1 import entry
        head,digest='a'*40,'b'*64
        train_raw,base_raw,params_raw=b'train',b'baseline',b'params'
        identity=dict(state_binding_sha256='s'*64,source_binding_sha256='n'*64,
            baseline_sha256=e._sha(base_raw),params_sha256=e._sha(params_raw),train_sha256=e._sha(train_raw),
            schema_sha256='c'*64,metadata_sha256='d'*64)
        binding=dict(schema='synthetic',salt='salt',**identity)
        docs={'claim.local.json':dict(schema='state-train-fit-claim-v1',commit=head,attempt=1,
                                      freeze_sha256=digest,retry_allowed=False),
              'attempt.local.json':dict(schema='state-train-fit-attempt-v1',commit=head,retry_allowed=False),
              'owner.local.json':{}}
        payloads={'train.local.npz':train_raw,'baseline.local.npz':base_raw,'params.local.npz':params_raw}
        def read(path,*_args,**_kwargs):
            name=Path(path).name
            return payloads[name] if name in payloads else e.previous._encoded(docs[name])
        writes={}
        def write(path,value):writes[Path(path).name]=value
        network=SimpleNamespace(score=Mock(return_value=np.array([1.,0.] if mismatch else [0.,1.],np.float64)),
                                mean=np.zeros(373,np.float64),scale=np.ones(373,np.float64),metadata={})
        arrays=old_test.source_arrays('train')
        def assess(groups,students,roles,**_kwargs):
            self.assertEqual(len(groups),42)
            self.assertTrue(all(group.source=='new' for group in groups))
            self.assertEqual((roles.count('A03'),roles.count('A04')),(21,21))
            self.assertEqual(len(students),42)
            return metrics()
        ops=(patch.object(e,'_runtime_binding',return_value=(head,binding,digest)),
             patch.object(e,'_read',side_effect=read),patch.object(e.previous,'_write_new',side_effect=write),
             patch.object(e.previous,'_guard'),patch.object(Path,'exists',return_value=False),
             patch.object(membership,'verify_worker'),patch.object(entry,'verify_live_owner'),
             patch.dict(e.os.environ,{membership.ENV_KEY:'synthetic-job'}),
             patch.object(e,'_source_contracts',return_value=identity),
             patch.object(e,'_train_arrays',return_value=arrays),
             patch.object(e,'_models',return_value=(network,object())),
             patch.object(analysis,'assess',side_effect=assess))
        with ExitStack() as stack:
            for op in ops:stack.enter_context(op)
            try:e.child()
            except ValueError:
                if not mismatch:raise
        return network,writes

    def test_only_training_groups_scored_and_result_written(self):
        network,writes=self.exercise()
        self.assertEqual(network.score.call_count,42)
        self.assertIn('worker-result.local.json',writes)
        self.assertNotIn('worker-failed.local.json',writes)

    def test_student_mismatch_fails_before_result(self):
        network,writes=self.exercise(mismatch=True)
        self.assertEqual(network.score.call_count,1)
        self.assertNotIn('worker-result.local.json',writes)
        self.assertIn('worker-failed.local.json',writes)


class SealTests(unittest.TestCase):
    @staticmethod
    def fixture():
        head,digest='a'*40,'b'*64
        result=fake_result(head)
        binding={key:result[key] for key in ('baseline_sha256','params_sha256','train_sha256')}
        encode=e.previous._encoded
        files={'worker-result.local.json':encode(result),'job.local.json':encode({'synthetic':True}),
            'attempt.local.json':encode(dict(schema='state-train-fit-attempt-v1',commit=head,retry_allowed=False)),
            'child-attempt.local.json':encode(dict(schema='state-train-fit-child-attempt-v1',commit=head,retry_allowed=False)),
            'claim.local.json':encode(dict(schema='state-train-fit-claim-v1',commit=head,attempt=1,
                                           freeze_sha256=digest,job_token_sha256='c'*64,retry_allowed=False)),
            'owner.local.json':encode(dict(schema='task-pair-fresh-multi-owner-v1',registration_commit=head,
                                           binding_sha256=digest,pid=1,created_us=1,deadline=100.0))}
        names=('worker-result','job','attempt','owner','claim','child-attempt')
        closed=dict(schema='state-train-fit-closed-v1',registration_commit=head,binding_sha256=digest,
                    receipts={name:e._sha(files[name+'.local.json']) for name in names},retry_allowed=False)
        files['closed.local.json']=encode(closed)
        return SimpleNamespace(binding=binding,digest=digest,result=result,files=files)

    def load(self,case,job_ok=True):
        def read(path,limit=None):
            name=Path(path).name
            if name not in case.files:raise FileNotFoundError('synthetic missing receipt')
            return case.files[name]
        with ExitStack() as stack:
            for op in (patch.object(e,'PRIVATE',Path('synthetic-stage')),
                    patch.object(e,'_read',side_effect=read),
                    patch.object(e,'_registered_binding',return_value=(case.binding,case.digest)),
                    patch.object(e.transfer,'_job_ok',return_value=job_ok),
                    patch.object(Path,'exists',lambda path:Path(path).name in case.files)):
                stack.enter_context(op)
            return e.load_completed(guard=guard)

    def test_complete_seal_and_bad_job_or_tampering(self):
        case=self.fixture();self.assertEqual(self.load(case),case.result)
        with self.assertRaises(ValueError):self.load(case,job_ok=False)
        for name in ('worker-result.local.json','job.local.json','claim.local.json'):
            case=self.fixture();case.files[name]+=b'x'
            with self.subTest(name=name),self.assertRaises(ValueError):self.load(case)
        case=self.fixture();case.files['worker-failed.local.json']=b'{}'
        with self.assertRaises(ValueError):self.load(case)


class ParentTests(unittest.TestCase):
    def exercise(self,job_ok):
        from learning.task_pair_fresh_multi_v1 import entry
        import psutil
        head,digest='a'*40,'b'*64
        binding={'baseline_sha256':'c'*64,'params_sha256':'d'*64,'train_sha256':'e'*64}
        files={}
        def write(path,value):
            name=Path(path).name
            if name in files:raise FileExistsError('synthetic marker')
            files[name]=e.previous._encoded(value)
        def read(path,limit=None):return files[Path(path).name]
        def supervise(command,_cwd,limits,name):
            self.assertEqual(command[1:],['-B','-m','learning.task_state_train_fit_v1.experiment','child'])
            self.assertTrue(name.startswith('Local\\TeacherDeviation-'))
            self.assertEqual(limits,e.JOB)
            files['worker-result.local.json']=b'{}';files['child-attempt.local.json']=b'{}'
            return dict(passed=job_ok)
        operations=(patch.object(e,'PRIVATE',Path('synthetic-stage')),
            patch.object(e.previous,'physical_available',return_value=8*1024**3),
            patch.object(e.shutil,'disk_usage',return_value=SimpleNamespace(free=8*1024**3)),
            patch.object(e,'_runtime_binding',return_value=(head,binding,digest)),
            patch.object(e.previous,'_write_new',side_effect=write),patch.object(e,'_read',side_effect=read),
            patch.object(e.previous,'_guard'),patch.object(psutil,'Process',return_value=SimpleNamespace(create_time=lambda:1.0)),
            patch.object(entry,'owner_marker',return_value=b'{"synthetic":true}'),
            patch.object(e.supervisor,'_supervise_owned',side_effect=supervise),
            patch.object(e.transfer,'_job_ok',side_effect=lambda job:job['passed']),
            patch.object(e,'_validated_outputs',return_value={}),
            patch.object(e,'_registered_binding',return_value=(binding,digest)))
        with ExitStack() as stack:
            for op in operations:stack.enter_context(op)
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:e.run()
            except RuntimeError:
                if job_ok:raise
        return files

    def test_job_failure_cannot_seal(self):
        passed=self.exercise(True);self.assertIn('closed.local.json',passed)
        failed=self.exercise(False)
        self.assertNotIn('closed.local.json',failed);self.assertIn('failed.local.json',failed)
