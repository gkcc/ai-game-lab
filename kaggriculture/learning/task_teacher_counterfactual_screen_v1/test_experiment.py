"""Pure synthetic result and failure checks, with no historical inputs."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase,main,mock

from learning.task_pair_fresh_multi_v1 import entry as owner_entry
from learning.task_student_dagger_collection_v1 import sources as dagger_sources
from . import experiment,screen


def case(expected, found):
    role,slot,opponent,side=expected
    plan=[[dict(step=0,ordinal=0)] if found else []]+[[] for _ in range(5)]
    statuses=[['official_next_frame_different','empty'] if found else ['empty','empty']]
    statuses.extend([['empty','empty'] for _ in range(5)])
    coverage=dict(role=role,opponent=opponent,side=side,status_by_window=statuses,
        ranked_out_by_window=[0]*6,source_disagreements_beyond_top4='unknown',
        comparisons=int(found),official_transitions_reserved=2*int(found),candidate_found=found)
    candidate=(dict(role=role,slot=slot,opponent=opponent,side=side,window=0,rank=0,
        step=0,ordinal=0,input_sha256='a'*64,original_choice='student',teacher_choice='teacher') if found else None)
    return screen.PrivateCase(coverage,candidate,plan)


def assembled(n=4):
    flags=[i in (0,1,4,5) for i in range(8)] if n==4 else [i==0 for i in range(8)]
    cases=[case(spec,found) for spec,found in zip(experiment.CASES,flags)]
    meter=screen.Meter()
    for _ in range(sum(flags)):
        meter.reserve_comparison();meter.complete_comparison();meter.reserve_official();meter.complete_official()
    return experiment._documents('a'*40,cases,meter.snapshot())


class ResultTests(TestCase):
    def test_runtime_source_manifest_covers_frozen_components_and_rejects_changes(self):
        for name in dagger_sources.RUNTIME_COMPONENTS:
            self.assertIn('kaggriculture/'+name,experiment.SOURCE_NAMES)
        for name in ('learning/task_student_continuation_v1/sources.py',
                     'learning/panel_eval_v3/experiment.py','learning/panel_eval_v3/launch.py',
                     'learning/task_job_empty_exit_v1/sampling.py',
                     'learning/task_training_job_supervision_v1/job.py'):
            self.assertIn('kaggriculture/'+name,experiment.SOURCE_NAMES)
        versions=dict(python='3.11.5',numpy='1.26.0',psutil='6.0.0',
                      **{'kaggle-environments':'1.33.0'})
        binding=dict(schema='teacher-counterfactual-screen-binding-v1',salt='a'*64,
            roles={'A05':1,'A06':2},baseline_sha256='b'*64,crop_sha256='c'*64,
            schema_sha256='d'*64,metadata_sha256='e'*64,software=versions)
        raw=experiment.previous._encoded(binding)
        key='kaggriculture/learning/task_dagger_once_v1/policy.py'
        frozen={key:'f'*64}
        registered=experiment.previous._encoded(experiment._registration(frozen,experiment._sha(raw),versions))
        with mock.patch.object(experiment,'_git',return_value=experiment.BASE_COMMIT), \
             mock.patch.object(experiment.subprocess,'check_output',return_value=registered), \
             mock.patch.object(experiment,'_read',return_value=raw), \
             mock.patch.object(experiment,'_source_hashes',return_value=frozen) as hashes:
            self.assertEqual(experiment._registered_binding('a'*40)[0],binding)
            hashes.return_value={key:'0'*64}
            with self.assertRaises(ValueError): experiment._registered_binding('a'*40)

    def test_versions_are_fixed_and_reject_paths_or_missing_values(self):
        versions=dict(python='3.11.5',numpy='1.26.0',psutil='6.0.0',
                      **{'kaggle-environments':'1.33.0'})
        self.assertEqual(experiment._validate_software(versions),versions)
        for mutated in (dict(versions,python='private/path'),
                        {k:v for k,v in versions.items() if k!='psutil'}):
            with self.assertRaises(ValueError): experiment._validate_software(mutated)

    def test_installed_engine_bytes_are_checked_even_without_official_calls(self):
        rules,core=b'synthetic rules',b'synthetic core'
        module=SimpleNamespace(ENGINE_SHA=experiment._sha(rules),ENGINE_CORE_SHA256=experiment._sha(core),
            rules=SimpleNamespace(__file__='synthetic_rules'),official_core=SimpleNamespace(__file__='synthetic_core'))
        def read(path,limit):
            return rules if path.name=='synthetic_rules' else core
        with mock.patch.object(experiment,'ENGINE_SHA',module.ENGINE_SHA), \
             mock.patch.object(experiment,'CORE_SHA',module.ENGINE_CORE_SHA256), \
             mock.patch.object(experiment,'_read',side_effect=read):
            experiment._engine_identity(module,lambda:None)
            core=b'modified core'
            with self.assertRaises(ValueError): experiment._engine_identity(module,lambda:None)

    def test_distinct_cases_and_both_roles_required(self):
        plan,result=assembled()
        self.assertEqual(result['gate']['status'],'sufficient_for_separate_design')
        self.assertEqual((result['gate']['candidate_cases'],result['gate']['roles_covered']),
                         (4,['A05','A06']))
        self.assertEqual(experiment._validate_documents('a'*40,plan,result),result)
        _plan,insufficient=assembled(1)
        self.assertEqual(insufficient['gate']['status'],'insufficient')

    def test_result_tamper_and_private_candidate_mismatch_rejected(self):
        plan,result=assembled()
        altered=deepcopy(result);altered['gate']['status']='passed'
        with self.assertRaises(ValueError): experiment._validate_documents('a'*40,plan,altered)
        altered=deepcopy(result);altered['meter']['official_transitions_confirmed_completed']-=2
        with self.assertRaises(ValueError): experiment._validate_documents('a'*40,plan,altered)
        altered=deepcopy(plan);altered['cases'][0]['candidate']['ordinal']=5
        with self.assertRaises(ValueError): experiment._validate_documents('a'*40,altered,result)

    def test_failed_job_never_seals_or_retries(self):
        writes={}
        owner=dict(schema='task-pair-fresh-multi-owner-v1',registration_commit='a'*40,
                   binding_sha256='b'*64,pid=1,created_us=1,deadline=1.0)
        with mock.patch.object(experiment.os,'name','nt'), \
             mock.patch.object(experiment.previous,'physical_available',return_value=8*1024**3), \
             mock.patch.object(experiment.shutil,'disk_usage',return_value=SimpleNamespace(free=8*1024**3)), \
             mock.patch.object(experiment,'_runtime_binding',return_value=('a'*40,{},'b'*64)), \
             mock.patch.object(experiment.previous,'_write_new',side_effect=lambda p,v:writes.setdefault(p.name,v)), \
             mock.patch.object(owner_entry,'owner_marker',return_value=experiment.previous._encoded(owner)), \
             mock.patch('psutil.Process',return_value=SimpleNamespace(create_time=lambda:1)), \
             mock.patch.object(experiment.supervisor,'_supervise_owned',return_value={}), \
             mock.patch.object(experiment.transfer,'_job_ok',return_value=False):
            with self.assertRaises(RuntimeError): experiment.run()
        self.assertIn('failed.local.json',writes)
        self.assertNotIn('closed.local.json',writes)
        self.assertIs(writes['failed.local.json']['retry_allowed'],False)


if __name__=='__main__': main()
