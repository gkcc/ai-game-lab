"""Synthetic-only entry checks. No source, model, real Job or training."""
from contextlib import ExitStack
from copy import deepcopy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from learning.task_multisource_rank_v1 import core as rank
from learning.task_multisource_rank_v1.dedup import Item
from . import core, experiment as e


def no_guard():
    pass


def candidate_rows():
    data = np.zeros((2, 98), np.float32)
    data[0, rank.OP_OFFSET] = 1
    data[1, rank.OP_OFFSET + 1] = 1
    return data


def artificial_example(pool, role, value, *, step=None, student=None):
    shared = np.zeros(275, np.float32)
    shared[0] = value
    return e._Example(Item(shared, candidate_rows(), 0, pool, role), step,
                      0 if step is not None else None, student)


def old_group(value):
    ex = artificial_example('old_train', 'T01', value)
    return SimpleNamespace(shared=ex.item.shared, candidates=ex.item.candidates, target_index=0)


def source_arrays(split):
    roles = (0, 1) if split == 'train' else (2, 3)
    each = 21 if split == 'train' else 24
    n = 2 * each
    shared = np.zeros((n, 275), np.float32)
    shared[:, 0] = np.arange(n, dtype=np.float32) + (1 if split == 'train' else 100)
    return dict(shared=shared, candidates=np.tile(candidate_rows(), (n, 1)),
        offsets=np.arange(n + 1, dtype=np.int32) * 2,
        teacher_target=np.zeros(n, np.int32), student_index=np.ones(n, np.int32),
        role=np.repeat(np.asarray(roles, np.int8), each),
        step=np.asarray([(i % 6) * 120 + i // 6 for _role in roles for i in range(each)], np.int16),
        ordinal=np.zeros(n, np.int16))


def all_examples():
    train = [old_group(1000 + i) for i in range(512)]
    diag = [old_group(2000 + i) for i in range(256)]
    return e._examples(train, diag, ['T01'] * 128 + ['T02'] * 128,
                       [source_arrays('train'), source_arrays('diagnostic')], no_guard)


def bucket(n):
    return dict(groups=n, coverage='observed' if n else 'unknown', baseline_correct=0,
                residual_correct=0, changed=0, op_flips=0, improved=0, regressed=0,
                baseline_nll=1.0 if n else None, residual_nll=1.0 if n else None)


def assessment(roles, per_role):
    total = sum(per_role)
    def operations(n): return {op: bucket(n if op == 'DIG' else 0) for op in e.OPS}
    return dict(overall=bucket(total), by_target_operation=operations(total),
        by_role={role: dict(overall=bucket(n), by_target_operation=operations(n))
                 for role, n in zip(roles, per_role)}, independent_evaluation=False)


def fake_coverage():
    incoming = dict(old_train=512, new_train=42, old_diag=256, new_dev=48)
    retained = dict(old_train=16, new_train=16, old_diag=32, new_dev=16)
    zero = dict.fromkeys(e.POOLS, 0)
    split = dict(input_count=858, retained_count=80, input_by_pool=incoming,
        retained_by_pool=retained.copy(), duplicate_removed_by_pool={k:incoming[k]-retained[k] for k in e.POOLS},
        conflict_removed_by_pool=zero.copy(), diagnostic_excluded_train_connected_by_pool=zero.copy(),
        train_component_count=32, all_feature_component_count=80, train_conflict_components=0,
        diagnostic_only_components=48, diagnostic_conflict_components=0, diagnostic_train_connected_components=0)
    return dict(split=split, filtered={k:dict(single=0, identical_rows=0, student_same=0) for k in e.POOLS},
        eligible_by_pool=retained.copy(), new_roles=dict(A03=8,A04=8,A09=8,A10=8),
        new_role_windows=dict(A03=3,A04=3,A09=3,A10=3),
        checks=dict(old_train=True,new_train=True,old_diag=True,new_dev=True,
                    each_new_role=True,each_new_role_windows=True), passed=True)


def fake_precision(n):
    return dict(groups=n, probability_max_abs_error=0.0, target_nll_max_abs_error=0.0,
                target_nll_max_limit_ratio=0.0, argmax_stable=True, passed=True)


def fake_result(head='a'*40):
    new_dev, old_diag = assessment(('A09','A10'), (8,8)), assessment(('T01','T02'), (16,16))
    return dict(schema='state-residual-result-v1', registration_commit=head, coverage=fake_coverage(),
        train_student_checked=dict(groups=42,mismatches=0), diagnostic_student_checked=dict(groups=48,mismatches=0),
        train_precision=fake_precision(32), diagnostic_precision=fake_precision(48),
        parity=dict(groups=80,allclose=True,baseline_argmax_mismatches=0,training_choice_mismatches=0,
            maximum_absolute_error=0.0,fixed_batch=16,fixed_width=256,train_width=2,
            train_width_groups=80,full_width_groups=80,groups_by_width={'2':80,'256':80},device='cpu'),
        new_dev=new_dev,old_diag=old_diag,screen=e._screen(new_dev,old_diag),
        baseline_sha256='b'*64,params_sha256='c'*64,updates_completed=64,
        nonzero_gradient_seen=True,output_layer_changed=True,device='cpu',new_games=0,gpu=False,
        independent_evaluation=False,promotion=False,
        software=dict(python='3.12',numpy='1.0',jax='1.0',optax='1.0'))


class EntryTests(unittest.TestCase):
    def test_source_contract_rejects_model_or_schema_mismatch(self):
        from learning.task_student_anchor_collection_v1 import sources
        metadata={'synthetic':True}
        prior={'baseline_sha256':'b'*64,'schema_sha256':'s'*64,
               'historical':{'schema_sha256':'h'*64}}
        old_binding={'earlier':prior}
        bound={'baseline_sha256':'b'*64,'crop_sha256':'c'*64,'schema_sha256':'s'*64}
        new_binding=dict(schema='synthetic',salt='private',**bound)
        completed=dict(registration_commit=e.NEW_COMMIT,
            summary=dict(data_gate='passed',retained_by_role=dict(A03=21,A04=21,A09=24,A10=24)),
            train_sha256='t'*64,diagnostic_sha256='d'*64)
        context=SimpleNamespace(task_sha256='b'*64,crop_sha256='c'*64,
                                model_normalization=lambda:(None,None,metadata))
        with patch.object(e.new,'load_completed',return_value=completed), \
             patch.object(e.old,'_registered_binding',return_value=(old_binding,'o'*64)), \
             patch.object(e.new,'_registered_binding',return_value=(new_binding,'n'*64)), \
             patch.object(sources,'open_context',return_value=context), \
             patch.object(e.new,'_context_binding',return_value=bound), \
             patch.object(e.cross,'_schema_identity',return_value='s'*64):
            identity,returned,_,_=e._source_contracts(no_guard)
            self.assertIs(returned,context)
            self.assertEqual(identity['baseline_sha256'],context.task_sha256)
            prior['baseline_sha256']='x'*64
            with self.assertRaises(ValueError):e._source_contracts(no_guard)
            prior['baseline_sha256']='b'*64
            prior['schema_sha256']='z'*64
            with self.assertRaises(ValueError):e._source_contracts(no_guard)

    def test_population_mapping_is_private_and_split_gate_holds(self):
        examples = all_examples()
        self.assertEqual(len(examples), 858)
        self.assertEqual([sum(ex.item.pool == pool for ex in examples) for pool in e.POOLS], [512,42,256,48])
        pools, coverage = e._filter_pools(examples, np.zeros(373,np.float64), np.ones(373,np.float64), no_guard)
        e._validate_coverage(coverage)
        self.assertTrue(coverage['passed'])
        self.assertEqual([len(pools[key]) for key in e.POOLS], [512,42,256,48])
        self.assertEqual(coverage['new_role_windows'], dict(A03=6,A04=6,A09=6,A10=6))
        self.assertNotIn('array', repr(examples[0]))
        self.assertFalse(examples[0].item.shared.flags.writeable)

    def test_score_verifies_student_without_passing_label_or_metadata(self):
        ex = artificial_example('new_train','A03',1,step=0,student=1)
        network = SimpleNamespace(score=Mock(return_value=np.array([0.,1.],np.float64)))
        result = e._score([ex], network, verify_students=True, guard=no_guard)
        self.assertEqual(result[id(ex.item)].target, 0)
        self.assertEqual(len(network.score.call_args.args), 2)
        network.score = Mock(return_value=np.array([1.,0.],np.float64))
        with self.assertRaises(ValueError): e._score([ex],network,verify_students=True,guard=no_guard)

    def test_precision_centering_probability_and_target_nll(self):
        ex = artificial_example('new_train','A03',1,step=0,student=1)
        group = rank.Group(ex.item.shared,ex.item.candidates,np.array([1000.,1001.],np.float64),0,'new')
        report=e._precision([group],no_guard)
        self.assertTrue(report['passed'])
        self.assertLessEqual(report['probability_max_abs_error'], 1e-6)
        self.assertLessEqual(report['target_nll_max_limit_ratio'], 1)
        bad=rank.Group(ex.item.shared,ex.item.candidates,np.array([-1e-50,0.],np.float64),0,'new')
        with self.assertRaises(e.PrecisionError) as caught:e._precision([bad],no_guard)
        self.assertEqual(caught.exception.reason,'centered_cast_or_argmax')
        target_group=rank.Group(ex.item.shared,ex.item.candidates,np.array([0.,-20.],np.float64),1,'new')
        with patch.object(core,'_training_baseline',return_value=(np.array([0.,-20.]),np.array([0.,-20.001],np.float32))):
            with self.assertRaises(e.PrecisionError) as caught:e._precision([target_group],no_guard)
        self.assertEqual(caught.exception.reason,'target_nll')
        with patch.object(core,'_training_baseline',return_value=(np.array([0.,-1.]),np.array([0.,-1.01],np.float32))):
            with self.assertRaises(e.PrecisionError) as caught:e._precision([target_group],no_guard)
        self.assertEqual(caught.exception.reason,'probability')

    def test_parameter_archive_is_bounded_and_nonzero(self):
        base = core.initial_params(7)
        arrays = {key: getattr(base,key).copy() for key in ('ws','wc','b','v')}
        arrays['v'][0] = 1
        buf=io.BytesIO();np.savez_compressed(buf,**arrays)
        restored=e._decode_params(buf.getvalue())
        np.testing.assert_array_equal(restored.v,arrays['v'])
        arrays['v'][:] = 0
        buf=io.BytesIO();np.savez_compressed(buf,**arrays)
        with self.assertRaises(ValueError):e._decode_params(buf.getvalue())

    def test_result_accounting_and_parity_tamper_fail_closed(self):
        result=fake_result()
        self.assertEqual(e._result(e.previous._encoded(result),'a'*40)['screen']['screen'],'failed')
        for mutation in ('parity_count','precision','coverage','roles','student'):
            bad=deepcopy(result)
            if mutation=='parity_count':bad['parity']['train_width_groups']=31
            if mutation=='precision':bad['train_precision']['target_nll_max_limit_ratio']=1.1
            if mutation=='coverage':bad['coverage']['eligible_by_pool']['new_train']-=1
            if mutation=='roles':bad['coverage']['new_roles']['A09']-=1
            if mutation=='student':bad['diagnostic_student_checked']['mismatches']=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                e._result(e.previous._encoded(bad),'a'*40)

    def test_coverage_rejects_more_samples_than_role_window_capacity(self):
        coverage=fake_coverage()
        e._validate_coverage(coverage)
        coverage['new_roles'].update(A03=16,A04=4)
        coverage['eligible_by_pool']['new_train']=20
        coverage['split']['retained_by_pool']['new_train']=20
        coverage['split']['duplicate_removed_by_pool']['new_train']-=4
        coverage['split']['retained_count']+=4
        coverage['split']['train_component_count']+=4
        coverage['split']['all_feature_component_count']+=4
        with self.assertRaises(ValueError):e._validate_coverage(coverage)
        coverage=fake_coverage()
        coverage['new_role_windows']['A03']=0
        with self.assertRaises(ValueError):e._validate_coverage(coverage)

    def test_coverage_components_match_retained_groups(self):
        for key in ('train_component_count','diagnostic_only_components'):
            coverage=fake_coverage()
            coverage['split'][key]=0
            with self.subTest(key=key),self.assertRaises(ValueError):e._validate_coverage(coverage)
        coverage=fake_coverage()
        coverage['split']['train_conflict_components']=coverage['split']['train_component_count']+1
        with self.assertRaises(ValueError):e._validate_coverage(coverage)
        coverage=fake_coverage()
        coverage['split']['diagnostic_conflict_components']=-1
        with self.assertRaises(ValueError):e._validate_coverage(coverage)

    def test_assessment_rejects_impossible_improvements_or_regressions(self):
        for condition in ('no_baseline_correct','all_baseline_correct'):
            value=assessment(('A09','A10'),(8,8))
            all_buckets=[value['overall'],value['by_target_operation']['DIG']]
            for role in ('A09','A10'):
                section=value['by_role'][role]
                all_buckets.extend((section['overall'],section['by_target_operation']['DIG']))
            if condition=='all_baseline_correct':
                for row in all_buckets:
                    row['baseline_correct']=row['residual_correct']=row['groups']
            for row in (value['overall'],value['by_target_operation']['DIG'],
                        value['by_role']['A09']['overall'],value['by_role']['A09']['by_target_operation']['DIG']):
                row.update(changed=2,improved=1,regressed=1)
            with self.subTest(condition=condition),self.assertRaises(ValueError):
                e._validate_assessment(value,('A09','A10'),'new')

    def test_screen_treats_missing_old_coverage_as_unknown_unless_another_failure(self):
        new_dev=assessment(('A09','A10'),(8,8))
        old_diag=assessment(('T01','T02'),(2,2))
        # Set a real improvement in one role, with consistent aggregate counts.
        for row in (new_dev['overall'],new_dev['by_target_operation']['DIG'],
                    new_dev['by_role']['A09']['overall'],new_dev['by_role']['A09']['by_target_operation']['DIG']):
            row.update(residual_correct=2,changed=2,improved=2)
        result=e._screen(new_dev,old_diag)
        self.assertEqual(result['screen'],'unknown')
        self.assertIsNone(result['checks']['old_non_water_drop_at_most_five_points'])
        new_dev['overall']['changed']=0
        self.assertEqual(e._screen(new_dev,old_diag)['screen'],'failed')


class ChildOrderingTests(unittest.TestCase):
    def exercise(self, failure=None):
        from learning.task_teacher_dev_deviation_v1 import membership
        from learning.task_pair_fresh_multi_v1 import entry
        from . import evaluate
        head, binding_sha = 'a'*40, 'b'*64
        metadata = {'synthetic': True}
        baseline = b'synthetic baseline'
        identity = dict(metadata_sha256=e._sha(e.previous._encoded(metadata)),
                        baseline_sha256=e._sha(baseline),train_sha256='c'*64,diagnostic_sha256='d'*64)
        binding = dict(schema='binding',salt='salt',**identity)
        documents = {'claim.local.json':dict(schema='state-residual-claim-v1',commit=head,attempt=1,
                                             freeze_sha256=binding_sha,retry_allowed=False),
                     'attempt.local.json':dict(schema='state-residual-attempt-v1',commit=head,retry_allowed=False),
                     'owner.local.json':{}}
        new_train=[artificial_example('new_train','A03',i+1,step=i,student=1) for i in range(42)]
        new_dev=[artificial_example('new_dev','A09',i+100,step=i,student=1) for i in range(48)]
        old_train=[artificial_example('old_train','T01',i+200) for i in range(16)]
        old_diag=[artificial_example('old_diag','T01',i+300) for i in range(32)]
        examples=new_train+new_dev+old_train+old_diag
        pools={'old_train':old_train,'new_train':new_train[:16],'old_diag':old_diag,'new_dev':new_dev[:16]}
        coverage=fake_coverage()
        if failure == 'coverage': coverage['passed']=False
        context=SimpleNamespace(model_normalization=lambda:(np.zeros(373,np.float64),np.ones(373,np.float64),metadata))
        events=[]; writes={}; fit_completed=[False]
        def read(path,*_args,**_kwargs):
            name=Path(path).name
            return baseline if name=='baseline.local.npz' else json.dumps(documents[name]).encode()
        def write(path,value): writes[Path(path).name]=value
        def collect(*_args):events.append('old_archives_complete');return [],[],[]
        def model(*_args):events.append('baseline_snapshot');return object()
        def score(items,_network,*,verify_students,guard):
            pool=items[0].item.pool
            events.append('score_'+pool)
            if pool=='new_dev':self.assertTrue(fit_completed[0])
            if failure=='train_student' and pool=='new_train':raise ValueError('synthetic mismatch')
            if failure=='diagnostic_student' and pool=='new_dev':raise ValueError('synthetic mismatch')
            return {id(ex.item):rank.Group(ex.item.shared,ex.item.candidates,np.array([0.,1.],np.float64),
                                           ex.item.target,'new' if pool.startswith('new') else 'old') for ex in items}
        def fit(*_args,**_kwargs):
            events.append('fit');fit_completed[0]=True
            return SimpleNamespace(updates_completed=64,nonzero_gradient_seen=True,
                                   output_layer_changed=True,params=object())
        def parity(groups,*_args,**kwargs):
            events.append('parity')
            return dict(train_width_groups=len(groups),full_width_groups=len(groups))
        jax=SimpleNamespace(devices=lambda _kind:[SimpleNamespace(platform='cpu')],__version__='synthetic')
        optax=SimpleNamespace(__version__='synthetic')
        source_result=dict(summary=dict(data_gate='passed'),train_sha256='c'*64,diagnostic_sha256='d'*64)
        operations=(
            patch.object(e,'_runtime_binding',return_value=(head,binding,binding_sha)),
            patch.object(e,'_read',side_effect=read),patch.object(e.previous,'_write_new',side_effect=write),
            patch.object(e.previous,'_guard'),patch.object(Path,'exists',return_value=False),
            patch.object(membership,'verify_worker'),patch.object(entry,'verify_live_owner'),
            patch.dict(e.os.environ,{membership.ENV_KEY:'synthetic-job'}),
            patch.object(e,'_source_contracts',return_value=(identity,context,{}, {'operation_columns':[0,1,2,3,4]})),
            patch.object(e.new,'_validated_outputs',return_value=(source_result,[])),
            patch.object(e.old,'_collect_old',side_effect=collect),
            patch.object(e,'_examples',return_value=examples),patch.object(e,'_filter_pools',return_value=(pools,coverage)),
            patch.object(e,'_validate_coverage'),patch.object(e,'_model',side_effect=model),
            patch.object(e,'_score',side_effect=score),patch.object(e,'_precision',return_value={'passed':True}),
            patch.object(core,'fit',side_effect=fit),patch.object(e,'_write_params',return_value=(object(),'e'*64)),
            patch.object(evaluate,'parity',side_effect=parity),patch.object(evaluate,'assess',return_value={}),
            patch.object(e,'_screen',return_value={'screen':'failed'}),patch.object(e,'_result'),
            patch.dict('sys.modules',{'jax':jax,'optax':optax}))
        with ExitStack() as stack:
            for operation in operations:stack.enter_context(operation)
            try:e.child()
            except ValueError:
                if failure is None:raise
        return events,writes

    def test_train_students_before_fit_and_diagnostics_only_after_fit(self):
        events,writes=self.exercise()
        self.assertLess(events.index('old_archives_complete'),events.index('score_new_train'))
        self.assertLess(events.index('score_new_train'),events.index('fit'))
        self.assertLess(events.index('fit'),events.index('score_new_dev'))
        self.assertLess(events.index('score_new_dev'),events.index('parity'))
        self.assertIn('worker-result.local.json',writes)
        self.assertNotIn('worker-failed.local.json',writes)

    def test_failure_never_writes_complete_result(self):
        for stage in ('coverage','train_student','diagnostic_student'):
            events,writes=self.exercise(stage)
            self.assertNotIn('worker-result.local.json',writes)
            self.assertIn('worker-failed.local.json',writes)
            if stage in ('coverage','train_student'):self.assertNotIn('fit',events)
            if stage=='diagnostic_student':self.assertIn('fit',events)


class SealTests(unittest.TestCase):
    @staticmethod
    def fixture():
        head, binding_sha = 'a'*40, 'b'*64
        baseline=b'synthetic baseline'
        params=core.initial_params(8)
        arrays={key:getattr(params,key).copy() for key in ('ws','wc','b','v')}
        arrays['v'][0]=1
        buf=io.BytesIO();np.savez_compressed(buf,**arrays)
        raw_params=buf.getvalue()
        result=fake_result(head)
        result['baseline_sha256']=e._sha(baseline)
        result['params_sha256']=e._sha(raw_params)
        encode=e.previous._encoded
        files={
            'worker-result.local.json':encode(result),
            'coverage.local.json':encode(result['coverage']),
            'job.local.json':encode({'synthetic':True}),
            'attempt.local.json':encode(dict(schema='state-residual-attempt-v1',commit=head,retry_allowed=False)),
            'child-attempt.local.json':encode(dict(schema='state-residual-child-attempt-v1',commit=head,retry_allowed=False)),
            'claim.local.json':encode(dict(schema='state-residual-claim-v1',commit=head,attempt=1,
                                           freeze_sha256=binding_sha,job_token_sha256='c'*64,retry_allowed=False)),
            'owner.local.json':encode(dict(schema='task-pair-fresh-multi-owner-v1',registration_commit=head,
                                           binding_sha256=binding_sha,pid=1,created_us=1,deadline=100.0)),
            'baseline.local.npz':baseline,'params.local.npz':raw_params}
        names=('worker-result','coverage','job','attempt','owner','claim','child-attempt')
        closed=dict(schema='state-residual-closed-v1',registration_commit=head,binding_sha256=binding_sha,
                    receipts={name:e._sha(files[name+'.local.json']) for name in names},
                    baseline_sha256=result['baseline_sha256'],params_sha256=result['params_sha256'],retry_allowed=False)
        files['closed.local.json']=encode(closed)
        return SimpleNamespace(head=head,binding_sha=binding_sha,binding={'baseline_sha256':result['baseline_sha256']},
                               result=result,closed=closed,files=files,encode=encode)

    def load(self,case,job_ok=True):
        def read(path,limit=None):
            name=Path(path).name
            if name not in case.files:raise FileNotFoundError('synthetic missing receipt')
            return case.files[name]
        with ExitStack() as stack:
            for operation in (patch.object(e,'PRIVATE',Path('synthetic-stage')),
                    patch.object(e,'_read',side_effect=read),
                    patch.object(e,'_registered_binding',return_value=(case.binding,case.binding_sha)),
                    patch.object(e.transfer,'_job_ok',return_value=job_ok),
                    patch.object(Path,'exists',lambda path:Path(path).name in case.files)):
                stack.enter_context(operation)
            return e.load_completed(guard=no_guard)

    def test_complete_seal_authenticates_receipts_and_private_models(self):
        case=self.fixture()
        self.assertEqual(self.load(case),case.result)

    def test_failed_job_missing_or_tampered_receipt_and_params_are_rejected(self):
        case=self.fixture()
        with self.assertRaises(ValueError):self.load(case,job_ok=False)
        for name in ('coverage.local.json','worker-result.local.json','params.local.npz','baseline.local.npz'):
            case=self.fixture();case.files[name]+=b'x'
            with self.subTest(name=name),self.assertRaises(ValueError):self.load(case)
        case=self.fixture();case.files['worker-failed.local.json']=b'{}'
        with self.assertRaises(ValueError):self.load(case)
        case=self.fixture();case.closed['params_sha256']='d'*64
        case.files['closed.local.json']=case.encode(case.closed)
        with self.assertRaises(ValueError):self.load(case)


class ParentRunTests(unittest.TestCase):
    def exercise(self, job_passed):
        from learning.task_pair_fresh_multi_v1 import entry
        import psutil
        head, digest='a'*40,'b'*64
        binding={'baseline_sha256':'c'*64}
        result=dict(baseline_sha256='c'*64,params_sha256='d'*64,screen=dict(screen='failed'))
        files={}
        def write(path,value):
            name=Path(path).name
            if name in files:raise FileExistsError('synthetic exclusive marker')
            files[name]=e.previous._encoded(value)
        def read(path,limit=None):return files[Path(path).name]
        def supervise(command,_cwd,limits,name):
            self.assertEqual(command[1:],['-B','-m','learning.task_state_residual_v1.experiment','child'])
            self.assertEqual(e.os.environ['JAX_PLATFORMS'],'cpu')
            self.assertTrue(name.startswith('Local\\TeacherDeviation-'))
            self.assertEqual(limits,e.JOB)
            for filename in ('worker-result','coverage','child-attempt'):
                files[filename+'.local.json']=b'{"synthetic":true}'
            return dict(passed=job_passed)
        operations=(patch.object(e,'PRIVATE',Path('synthetic-stage')),
            patch.object(e.previous,'physical_available',return_value=8*1024**3),
            patch.object(e.shutil,'disk_usage',return_value=SimpleNamespace(free=8*1024**3)),
            patch.object(e,'_runtime_binding',return_value=(head,binding,digest)),
            patch.object(e.previous,'_write_new',side_effect=write),patch.object(e,'_read',side_effect=read),
            patch.object(e.previous,'_guard'),patch.object(psutil,'Process',return_value=SimpleNamespace(create_time=lambda:1.0)),
            patch.object(entry,'owner_marker',return_value=b'{"synthetic":true}'),
            patch.object(e.supervisor,'_supervise_owned',side_effect=supervise),
            patch.object(e.transfer,'_job_ok',side_effect=lambda job:job['passed']),
            patch.object(e,'_validated_outputs',return_value=result),
            patch.object(e,'_registered_binding',return_value=(binding,digest)))
        with ExitStack() as stack:
            for operation in operations:stack.enter_context(operation)
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:e.run()
            except RuntimeError:
                if job_passed:raise
        return files

    def test_success_closes_only_after_job_and_failed_job_has_no_seal(self):
        passed=self.exercise(True)
        self.assertIn('closed.local.json',passed)
        closed=e.previous._document(passed['closed.local.json'])
        for name in ('worker-result','coverage','job','attempt','owner','claim','child-attempt'):
            self.assertEqual(closed['receipts'][name],e._sha(passed[name+'.local.json']))
        failed=self.exercise(False)
        self.assertNotIn('closed.local.json',failed)
        self.assertIn('failed.local.json',failed)
