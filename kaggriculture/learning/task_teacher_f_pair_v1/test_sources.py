"""In-memory historical/candidate proofs; no private files or real seeds."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from learning.task_teacher_f_counterfactual_v1 import contract as c, worker
from learning.task_teacher_f_counterfactual_v1.test_contract import screen_fixture, hit
from learning.task_teacher_f_counterfactual_v1.test_sources import context_fixture
from . import sources as s, experiment as e


def public_fixture():
    old={f'synthetic/source-{i:03d}.py':f'original-{i}'.encode() for i in range(376)}
    old[s.PROGRESS]=b'original progress\n'
    reg=dict(schema='teacher-f-counterfactual-screen-prereg-v1',base_commit=c.BASE_COMMIT,
        engine_commit=c.ENGINE_COMMIT,engine_sha256=c.ENGINE_SHA,public_sources={p:s.sha(b) for p,b in old.items()},
        limits=c.LIMITS,screening=c.screening_contract(),phase_write_bytes=c.PHASE_BYTES,
        pair_execution_authorized=False,training=False,gpu=False,independent_evaluation=False,promotion=False,
        binding_commitment_sha256='b'*64)
    reg_raw=e.encoded(reg);summary_raw=e.encoded({'synthetic':True})
    cert=dict(schema='teacher-f-screen-result-certificate-v1',registration_commit=s.SCREEN_COMMIT,
        registration_sha256=s.sha(reg_raw),summary_sha256=s.sha(summary_raw),private_runtime_files=63,
        commitment_domain='teacher-f-screen-result-v1',private_runtime_commitment_sha256='a'*64,
        outcome='completed',accepted_sources=8,official_transitions=16)
    paths={s.SCREEN_EVIDENCE+'/registration.json':reg_raw,s.SCREEN_RESULTS+'/summary.json':summary_raw,
        s.SCREEN_RESULTS+'/certificate.json':e.encoded(cert)}
    current={**old,**paths};current[s.PROGRESS]+=b'published screen result\n'
    blobs={s.SCREEN_COMMIT+':'+p:b for p,b in old.items()}
    blobs[s.SCREEN_COMMIT+':'+s.SCREEN_EVIDENCE+'/registration.json']=reg_raw
    blobs.update({s.BASE+':'+p:b for p,b in paths.items() if p.startswith(s.SCREEN_RESULTS)})
    blobs[s.BASE+':'+s.PROGRESS]=current[s.PROGRESS]
    infra=SimpleNamespace(decode=json.loads,read=lambda p:current[p.relative_to(s.REPO).as_posix()])
    return infra,current,blobs,lambda command,path:blobs[path]


def completed_fixture():
    context=context_fixture();plans=context.private_source_plans();results=screen_fixture()
    for result,plan in zip(results,plans):
        hit(result)
        result['source_plan']=plan.document()
        result['coverage'].update(source_identity_sha256=plan.identity_sha256,
            selection_plan=[[dict(step=3,ordinal=1),None]]+[[None,None] for _ in range(5)])
        result['candidate']=dict(role=plan.role,slot=plan.slot,opponent=plan.opponent,side=plan.side,
            cohort=plan.cohort,alias=plan.alias,source_game=plan.source_game,source_identity_sha256=plan.identity_sha256,
            seed=plan.seed,task_sha256=plan.task_sha256,crop_sha256=plan.crop_sha256,window=0,rank=0,step=3,ordinal=1,
            input_sha256='a'*64,original_choice='synthetic-student',teacher_choice='synthetic-teacher')
    values={};default=b'synthetic default'
    binding=dict(schema='teacher-f-screen-binding-v1',base_commit=c.BASE_COMMIT,salt='d'*64,
        completed_sources=context.identities,source_plans=[p.document() for p in plans],default_sha256=s.sha(default))
    values['private/binding.local.json']=binding;binding_sha=s.sha(e.encoded(binding))
    values['private/attempt.local.json']=s._marker('attempt',binding_sha)
    values['private/owner.local.json']=dict(schema='task-pair-fresh-multi-owner-v1',registration_commit=s.SCREEN_COMMIT,
        binding_sha256=binding_sha,pid=100,created_us=1,deadline=1000.)
    accepted=[];jobs=[]
    for index,result in enumerate(results):
        prefix=f'screen/source-{index:02d}/'
        values[prefix+'readiness.local.json']=dict(consecutive_samples=3,last_available_bytes=8*1024**3,elapsed_seconds=20.)
        for kind in ('permit','started','consumed'):values[prefix+kind+'.local.json']=s._marker(kind,binding_sha,index)
        values[prefix+'result.local.json']=result
        job=dict(synthetic_job=True,launcher_reaped=True,job_members_zero=True,cleanup_error_kind=None)
        values[prefix+'supervision.local.json']=job;job_sha=s.sha(e.encoded(job))
        values[prefix+'accepted.local.json']=dict(schema='teacher-f-screen-accepted-v1',source_index=index,passed=True,
            registration_commit=s.SCREEN_COMMIT,binding_sha256=binding_sha,result_sha256=s.sha(e.encoded(result)),
            supervision_sha256=job_sha,retry_allowed=False)
        accepted.append(dict(source_index=index,sha256=s.sha(e.encoded(values[prefix+'accepted.local.json']))))
        jobs.append(dict(source_index=index,sha256=job_sha,launcher_reaped=True,job_members_zero=True,cleanup_error_kind=None))
    projection=c.screen_projection(results)
    values['private/screen-result.local.json']=dict(schema='teacher-f-screen-result-v1',registration_commit=s.SCREEN_COMMIT,
        binding_sha256=binding_sha,results=results,accepted=accepted,pair_execution_authorized=False)
    values['private/projected-summary.local.json']=projection
    values['private/screen-seal.local.json']=dict(schema='teacher-f-screen-seal-v1',registration_commit=s.SCREEN_COMMIT,
        binding_sha256=binding_sha,screen_result_sha256=s.sha(e.encoded(values['private/screen-result.local.json'])),
        projection_sha256=s.sha(e.encoded(projection)),accepted=accepted,pair_execution_authorized=False)
    values['private/closed.local.json']=dict(schema='teacher-f-screen-closed-v1',registration_commit=s.SCREEN_COMMIT,
        binding_sha256=binding_sha,outcome='completed',error_kind=None,accepted_sources=8,active_index=None,
        child_cleanup_receipts=jobs,retry_allowed=False)
    raws={key:e.encoded(value) for key,value in values.items()}
    files={key:dict(sha256=s.sha(raw),bytes=len(raw)) for key,raw in raws.items()}
    commitment=s.commitment(files,binding['salt'])
    witness=dict(schema='teacher-f-screen-publication-witness-v1',files=files,
        commitment_sha256=commitment,witness_excluded=True)
    paths={s._path(key):raw for key,raw in raws.items()}
    paths[s.PRIVATE/s.WITNESS]=e.encoded(witness);paths[s.GAME/'main.py']=default
    proof=dict(registration=dict(base_commit=c.BASE_COMMIT,binding_commitment_sha256=binding_sha),
        certificate=dict(private_runtime_commitment_sha256=commitment),summary=projection,identity={'synthetic':True})
    infra=SimpleNamespace(read=lambda p:paths[p],decode=json.loads)
    modules=dict(acceptance=SimpleNamespace(supervision_ok=lambda job:job.get('synthetic_job') is True),
        f_sources=SimpleNamespace(open_context=lambda **_kwargs:context),f_worker=worker)
    return infra,modules,proof,raws,paths,context,files


class SourceTests(unittest.TestCase):
    def test_cross_commit_public_proof_only_allows_suffix_after_latest_published_progress(self):
        infra,current,blobs,git=public_fixture()
        current[s.PROGRESS]+=b'new pair plan\n'
        self.assertEqual(s.public(infra,git,c)['identity']['critical_files_unchanged'],376)
        current[s.PROGRESS]=blobs[s.SCREEN_COMMIT+':'+s.PROGRESS]
        with self.assertRaises(ValueError):s.public(infra,git,c)
        current[s.PROGRESS]=blobs[s.BASE+':'+s.PROGRESS]
        current['synthetic/source-005.py']+=b'changed'
        with self.assertRaises(ValueError):s.public(infra,git,c)

    def test_published_registration_certificate_and_summary_are_preserved_on_disk(self):
        for relative in (s.SCREEN_EVIDENCE+'/registration.json',s.SCREEN_RESULTS+'/certificate.json',s.SCREEN_RESULTS+'/summary.json'):
            infra,current,_blobs,git=public_fixture();current[relative]+=b'\n'
            with self.subTest(relative=relative),self.assertRaises(ValueError):s.public(infra,git,c)

    def test_load_authenticates_all_63_files_and_all_eight_cases_without_screening_again(self):
        infra,modules,proof,raws,paths,context,files=completed_fixture()
        inventory=lambda root,prefix:({x for x in s.FILE_KEYS if x.startswith(prefix+'/')}
            |({'private/'+s.WITNESS} if prefix=='private' else set()))
        with patch.object(s,'_inventory',inventory):
            evidence=s.load(infra,modules,c,proof,lambda:None)
            self.assertEqual(len(evidence.cases),8);self.assertEqual(len(evidence.identity['files']),63)
            evidence.verify_files(infra,lambda:None)
            paths[s._path('screen/source-04/consumed.local.json')]+=b'\n'
            with self.assertRaises(ValueError):evidence.verify_files(infra,lambda:None)
            with self.assertRaises(ValueError):s.load(infra,modules,c,proof,lambda:None)
        evidence.close()
        self.assertNotIn('private/'+s.WITNESS,files)
        with self.assertRaises(ValueError):s.verify_manifest(dict(files,**{'private/'+s.WITNESS:{}}),raws,'d'*64,'a'*64)

    def test_fixed_source_order_rejects_missing_duplicate_empty_and_foreign_case(self):
        _infra,_modules,_proof,_raws,_paths,context,_files=completed_fixture()
        results=[dict(source=dict(src),candidate=dict(role=src['role'],opponent=src['opponent'],side=src['side'])) for src in c.SOURCES]
        self.assertEqual(len(s.ordered_cases(results,c)),8)
        broken=[deepcopy(results) for _ in range(5)]
        broken[0].pop();broken[1][1]=deepcopy(broken[1][0]);broken[2][0]['candidate']=None
        broken[3][0],broken[3][1]=broken[3][1],broken[3][0]
        broken[4][0]['candidate']['role']='A05'
        for value in broken:
            with self.assertRaises(ValueError):s.ordered_cases(value,c)
        context.close()


if __name__=='__main__':unittest.main()
