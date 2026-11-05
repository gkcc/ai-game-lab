"""Synthetic phase-chain regressions; the optional scratch test uses no games."""
from copy import deepcopy
import json
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack,contextmanager
from contextlib import redirect_stdout,redirect_stderr

from learning.task_teacher_counterfactual_pair_v1 import experiment as legacy
from learning.task_teacher_counterfactual_pair_v1.test_experiment import phase_result
from learning.task_teacher_f_counterfactual_v1 import contract as c
from . import experiment as e


def chain_fixture(count=56,*,binding_sha='b'*64):
    files={};infra=SimpleNamespace(decode=json.loads);owner=dict(deadline=10000.)
    common=dict(infra=infra,contract=c,pair=legacy,job_ok=lambda x:x=={'synthetic_job':True},
        commit='a'*40,binding_sha=binding_sha,owner=owner)
    previous=e.state()
    for sequence,unit in enumerate(e.units(c)[:count]):
        ready=e.encoded(dict(consecutive_samples=3,last_available_bytes=8*1024**3,elapsed_seconds=20.))
        claim=e.encoded(dict(schema='teacher-f-pair-claim-v1',registration_commit=common['commit'],
            binding_sha256=common['binding_sha'],sequence=sequence,unit=unit,attempt=1,retry_allowed=False,
            previous_sha256=previous['previous_sha256'],limit_bytes=c.PHASE_BYTES[unit['phase']],deadline=900.,readiness_sha256=e.sha(ready)))
        marker=e.encoded(dict(schema='teacher-f-pair-child-attempt-v1',registration_commit=common['commit'],
            binding_sha256=common['binding_sha'],sequence=sequence,unit=unit,claim_sha256=e.sha(claim),retry_allowed=False))
        storage=dict(limit_bytes=c.PHASE_BYTES[unit['phase']],used_bytes=3,remaining_bytes=c.PHASE_BYTES[unit['phase']]-3,refused=0,created_count=1)
        result=e.encoded(dict(schema='teacher-f-pair-child-result-v1',registration_commit=common['commit'],
            binding_sha256=common['binding_sha'],sequence=sequence,unit=unit,claim_sha256=e.sha(claim),
            result=phase_result(unit['phase']),storage=storage,retry_allowed=False))
        raws={'readiness':ready,'claim':claim,'child-attempt':marker,'child-result':result,'job':e.encoded({'synthetic_job':True})}
        previous,accepted=e.validate_phase(raws,sequence,previous,**common)
        for name,raw in raws.items():files[e.numbered(name,sequence)]=raw
        accepted_raw=e.encoded(accepted);files[e.numbered('accepted',sequence)]=accepted_raw
        previous['previous_sha256']=e.sha(accepted_raw)
    return files,common


class ChainTests(unittest.TestCase):
    def test_help_succeeds_without_entry_and_execution_system_exit_is_not_success(self):
        out,err=io.StringIO(),io.StringIO()
        with patch.object(e.sys,'argv',['synthetic','--help']),redirect_stdout(out),redirect_stderr(err):
            self.assertEqual(e.main(),0)
        self.assertFalse(err.getvalue())
        with patch.object(e.sys,'argv',['synthetic','--unknown','synthetic-private-value']),redirect_stdout(out),redirect_stderr(err):
            self.assertEqual(e.main(),1)
        self.assertNotIn('synthetic-private-value',err.getvalue())
        with patch.object(e.sys,'argv',['synthetic','--prepare']),patch.object(e,'prepare',side_effect=SystemExit(0)),redirect_stderr(err):
            self.assertEqual(e.main(),1)

    def test_all_fifty_six_phases_include_both_official_audits_before_complete_summary(self):
        files,common=chain_fixture();result=e.read_chain(56,files.__getitem__,**common)
        self.assertEqual(len(result['accounts']),8);self.assertEqual(len(result['outputs']),8)
        self.assertEqual(result['charged_bytes'],56*3)
        self.assertEqual(c.pair_projection(result['accounts'],list(c.SOURCES))['represented_seed_clusters'],4)
        self.assertEqual([u['phase'] for u in e.units(c)[:7]],list(c.PHASES))

    def test_intervention_cannot_pass_a_missing_control_audit_or_control_verify_receipt(self):
        for sequence in (2,3):
            files,common=chain_fixture(5);del files[e.numbered('accepted',sequence)]
            with self.subTest(sequence=sequence),self.assertRaises(KeyError):e.read_chain(4,files.__getitem__,**common)

    def test_tampered_job_result_previous_link_and_partial_game_fail_closed(self):
        for kind,mutation in (
            ('job',lambda value:value.update(synthetic_job=False)),
            ('child-result',lambda value:value['result']['details'].update(states=719)),
            ('claim',lambda value:value.update(previous_sha256=None)),
            ('accepted',lambda value:value.update(charged_bytes=0))):
            files,common=chain_fixture(2);name=e.numbered(kind,1);value=json.loads(files[name]);mutation(value);files[name]=e.encoded(value)
            with self.subTest(kind=kind),self.assertRaises(ValueError):e.read_chain(2,files.__getitem__,**common)

    def test_bound_case_order_rejects_omission_duplicate_or_skipped_control_verification(self):
        cases=[dict(role=s['role'],opponent=s['opponent'],side=s['side']) for s in c.SOURCES]
        original=dict(schema='teacher-f-pair-binding-v1',base_commit=e.BASE,salt='a'*64,screen={'cases':cases},execution_order=e.units(c))
        e.binding_valid(original,c)
        bad=[deepcopy(original) for _ in range(4)]
        bad[0]['screen']['cases'].pop();bad[1]['screen']['cases'][1]=dict(cases[0])
        bad[2]['execution_order'].pop(3);bad[3]['execution_order'][3],bad[3]['execution_order'][4]=bad[3]['execution_order'][4],bad[3]['execution_order'][3]
        for value in bad:
            with self.assertRaises(ValueError):e.binding_valid(value,c)

    def test_parent_metadata_charges_exact_encoded_newline_and_does_not_refund_failed_write(self):
        infra=SimpleNamespace(write=lambda *_args:(_ for _ in ()).throw(OSError('synthetic failure')))
        payload={'safe':True};size=len(e.encoded(payload));budget=e.Metadata(infra,size)
        with self.assertRaises(OSError):budget.write(Path('synthetic-uncreated'),payload)
        self.assertEqual(budget.used,size)
        with self.assertRaises(ValueError):budget.write(Path('synthetic-uncreated-2'),payload)

    def test_original_nul_exception_preserves_file_write_budget_and_restores_os_open(self):
        root=os.environ.get('F_PAIR_SYNTHETIC_SCRATCH')
        if os.name!='nt' or not root:self.skipTest('Requires an explicitly owned synthetic scratch directory')
        root=Path(root);self.assertTrue((root/'.agent-workflow-owner.json').is_file())
        stage=root/'budget';stage.mkdir()
        from learning.task_anchor_once_v1.storage import OutputBudgetExceeded,UnsafeWriteError
        original=os.open
        with self.assertRaises(OutputBudgetExceeded):
            with legacy._owned_writes_with_subprocess_null(stage,5) as budget:
                descriptor=os.open('nul',os.O_RDWR);os.close(descriptor)
                with self.assertRaises(UnsafeWriteError):os.open(str(stage/'unmetered'),os.O_CREAT|os.O_WRONLY)
                with self.assertRaises(UnsafeWriteError):os.open('nul',os.O_WRONLY)
                (stage/'first').write_bytes(b'abc')
                with self.assertRaises(OutputBudgetExceeded):(stage/'second').write_bytes(b'xyz')
        self.assertEqual(budget.snapshot()['used_bytes'],3);self.assertEqual(budget.snapshot()['refused'],1)
        self.assertIs(os.open,original);self.assertFalse((stage/'unmetered').exists())

    def test_native_output_boundary_rejects_private_or_combined_directory_overflow(self):
        for private,stage in ((c.LIMITS['private_result_bytes']+1,0),(1,c.LIMITS['pair_output_bytes'])):
            modules={'driver':SimpleNamespace(_owned_size=lambda p:private if p==e.PRIVATE else stage)}
            with self.assertRaises(ValueError):e.disk_boundary(modules,c.LIMITS,lambda:None)


class CoordinatorTests(unittest.TestCase):
    def test_child_entry_uses_selected_f_case_after_audits_and_persists_known_failure_budget(self):
        for fail in (False,True):
            cases=[dict(role=s['role'],opponent=s['opponent'],side=s['side']) for s in c.SOURCES]
            binding=dict(schema='teacher-f-pair-binding-v1',base_commit=e.BASE,salt='a'*64,
                screen={'cases':cases},execution_order=e.units(c))
            raw=e.encoded(binding);binding_sha=e.sha(raw);sequence=18
            prior,common=chain_fixture(sequence,binding_sha=binding_sha)
            current=e.read_chain(sequence,prior.__getitem__,**common);unit=e.units(c)[sequence]
            owner=dict(schema='task-pair-fresh-multi-owner-v1',registration_commit='a'*40,binding_sha256=binding_sha,
                pid=100,created_us=1,deadline=10000.)
            files={e.PRIVATE/name:value for name,value in prior.items()}
            files.update({e.PRIVATE/'binding.local.json':raw,e.PRIVATE/'owner.local.json':e.encoded(owner),
                e.PRIVATE/'attempt.local.json':e.encoded(e.attempt('a'*40,binding_sha))})
            ready=prior[e.numbered('readiness',0)];files[e.PRIVATE/e.numbered('readiness',sequence)]=ready
            limit=c.PHASE_BYTES[unit['phase']]
            claim=dict(schema='teacher-f-pair-claim-v1',registration_commit='a'*40,binding_sha256=binding_sha,
                sequence=sequence,unit=unit,attempt=1,retry_allowed=False,previous_sha256=current['previous_sha256'],
                limit_bytes=limit,deadline=900.,readiness_sha256=e.sha(ready))
            files[e.PRIVATE/e.numbered('claim',sequence)]=e.encoded(claim)
            def write(path,value,maximum=e.MAX_RECEIPT):
                self.assertNotIn(path,files);data=e.encoded(value);self.assertLessEqual(len(data),maximum)
                files[path]=data;return e.sha(data)
            def close():
                if fail:raise ValueError('synthetic close failure')
            context=object();evidence=SimpleNamespace(identity=binding['screen'],context=context,cases=cases,
                close=close,verify_files=lambda *_args:None)
            calls=[]
            def phase(**kwargs):
                calls.append(kwargs);self.assertIs(kwargs['context'],context)
                if fail:raise MemoryError('synthetic primary failure')
                return phase_result(kwargs['phase'])
            @contextmanager
            def writes(_directory,_limit):
                yield SimpleNamespace(snapshot=lambda:dict(limit_bytes=limit,used_bytes=3,remaining_bytes=limit-3,refused=0,created_count=1))
            infra=SimpleNamespace(read=lambda p,*_args:files[p],write=write,decode=json.loads,snapshot=lambda *_args:None,
                closing_memory=lambda error,latest,reader:dict(failure_boundary_observation=dict(latest),memory_at_close={'available_bytes':reader()}))
            reg=dict(binding_commitment_sha256=binding_sha,
                public_sources={'kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py':'c'*64})
            auth=(infra,reg,c,SimpleNamespace(verify_public=lambda **_kwargs:None),None,
                SimpleNamespace(physical_available=lambda:8*1024**3),SimpleNamespace(load=lambda *_args,**_kwargs:evidence),{})
            modules=dict(pair=legacy,acceptance=SimpleNamespace(supervision_ok=lambda x:x=={'synthetic_job':True}),
                f_worker=SimpleNamespace(run_phase=phase),fresh=SimpleNamespace(verify_live_owner=lambda *_args,**_kwargs:None),
                driver=SimpleNamespace(_owned_size=lambda _p:0))
            with ExitStack() as stack:
                for target,name,value in ((e,'authenticate',lambda _c:auth),(e,'runtime',lambda *_args:modules),
                    (e.os.path,'lexists',lambda _p:False),(e.shutil,'disk_usage',lambda _p:SimpleNamespace(free=100*1024**3)),
                    (e.time,'monotonic',lambda:0.),(legacy,'_verify_outputs',lambda *_args:None),
                    (legacy,'_owned_writes_with_subprocess_null',writes)):
                    stack.enter_context(patch.object(target,name,value))
                if fail:
                    with self.assertRaises(MemoryError):e.run('a'*40,child_sequence=sequence)
                else:self.assertIsNone(e.run('a'*40,child_sequence=sequence))
            self.assertEqual(len(calls),1);self.assertEqual(calls[0]['case'],cases[2]);self.assertEqual(calls[0]['phase'],'intervene')
            receipt=json.loads(files[e.PRIVATE/e.numbered('child-failed' if fail else 'child-result',sequence)])
            self.assertEqual(receipt['storage']['used_bytes'],3)
            if fail:
                self.assertEqual(receipt['error_kind'],'MemoryError');self.assertEqual(receipt['context_close_error_kind'],'ValueError')
                self.assertNotIn(e.PRIVATE/e.numbered('child-result',sequence),files)

    def simulate(self,*,failure=False,close_failure=False,closed_write_failure=False):
        cases=[dict(role=s['role'],opponent=s['opponent'],side=s['side']) for s in c.SOURCES]
        binding=dict(schema='teacher-f-pair-binding-v1',base_commit=e.BASE,salt='a'*64,
            screen={'cases':cases},execution_order=e.units(c))
        files={e.PRIVATE/'binding.local.json':e.encoded(binding)};calls=[]
        read=lambda p,*_args:files[Path(p)]
        def write(path,value,maximum=e.MAX_RECEIPT):
            path=Path(path);self.assertNotIn(path,files)
            if path==e.PRIVATE/'closed.local.json' and closed_write_failure:
                raise OSError('synthetic closed write failure')
            raw=e.encoded(value);self.assertLessEqual(len(raw),maximum);files[path]=raw;return e.sha(raw)
        def close():
            if close_failure:raise ValueError('synthetic close failure')
        evidence=SimpleNamespace(identity=binding['screen'],close=close,verify_files=lambda *_args:None)
        gate=dict(consecutive_samples=3,last_available_bytes=8*1024**3,elapsed_seconds=20.)
        job=dict(synthetic_job=True,launcher_reaped=True,job_members_zero=True,cleanup_error_kind=None)
        def supervised(_campaign,command,_cwd,_limits):
            sequence=int(command[-1]);calls.append(sequence)
            if failure:raise RuntimeError('synthetic original failure')
            claim_raw=read(e.PRIVATE/e.numbered('claim',sequence));claim=json.loads(claim_raw);unit=claim['unit']
            header=dict(registration_commit='a'*40,binding_sha256=e.sha(files[e.PRIVATE/'binding.local.json']),
                sequence=sequence,unit=unit,claim_sha256=e.sha(claim_raw),retry_allowed=False)
            write(e.PRIVATE/e.numbered('child-attempt',sequence),dict(schema='teacher-f-pair-child-attempt-v1',**header))
            limit=claim['limit_bytes'];storage=dict(limit_bytes=limit,used_bytes=3,remaining_bytes=limit-3,refused=0,created_count=1)
            write(e.PRIVATE/e.numbered('child-result',sequence),dict(schema='teacher-f-pair-child-result-v1',
                **header,result=phase_result(unit['phase']),storage=storage))
            return dict(job)
        infra=SimpleNamespace(read=read,write=write,decode=json.loads,
            snapshot=lambda *_args:SimpleNamespace(await_memory=lambda *_args,**_kwargs:dict(gate)),
            supervise_recorded=supervised,closing_memory=lambda error,latest,reader:dict(
                failure_boundary_observation=dict(latest) if error else None,memory_at_close=dict(available_bytes=reader())))
        reg=dict(binding_commitment_sha256=e.sha(files[e.PRIVATE/'binding.local.json']),
            public_sources={'kaggriculture/learning/task_teacher_fresh_panel_v1/contract.py':'c'*64})
        auth=(infra,reg,c,SimpleNamespace(verify_public=lambda **_kwargs:None),None,
            SimpleNamespace(physical_available=lambda:8*1024**3),SimpleNamespace(load=lambda *_args,**_kwargs:evidence),{})
        modules=dict(pair=legacy,acceptance=SimpleNamespace(supervision_ok=lambda x:x==job),campaign=None,
            driver=SimpleNamespace(_owned_size=lambda p:sum(len(v) for k,v in files.items() if k.is_relative_to(p))))
        with ExitStack() as stack:
            stack.enter_context(patch.object(e,'authenticate',return_value=auth))
            stack.enter_context(patch.object(e,'runtime',return_value=modules))
            stack.enter_context(patch.object(e.os.path,'lexists',return_value=False))
            stack.enter_context(patch.object(e.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)))
            stack.enter_context(patch.object(Path,'mkdir'))
            stack.enter_context(patch.object(legacy,'_verify_outputs',return_value=None))
            error=None;result=None
            try:result=e.run('a'*40)
            except BaseException as exc:error=exc
        return result,error,files,calls

    def test_parent_orchestrator_accepts_all_fifty_six_synthetic_jobs_before_summary(self):
        result,error,files,calls=self.simulate()
        self.assertIsNone(error);self.assertEqual(calls,list(range(56)));self.assertEqual(result['cases'],8)
        closed=json.loads(files[e.PRIVATE/'closed.local.json'])
        self.assertEqual(closed['outcome'],'completed');self.assertEqual(closed['accepted_phases'],56)
        self.assertEqual(closed['child_metadata_reserved_bytes'],56*3*e.MAX_RECEIPT)
        self.assertEqual(len(json.loads(files[e.PRIVATE/'pair-result.local.json'])['accounts']),8)

    def test_original_failure_survives_context_close_error_and_unknown_job_cleanup_stops(self):
        result,error,files,calls=self.simulate(failure=True,close_failure=True)
        self.assertIsNone(result);self.assertIsInstance(error,RuntimeError);self.assertEqual(calls,[0])
        closed=json.loads(files[e.PRIVATE/'closed.local.json'])
        self.assertEqual(closed['error_kind'],'RuntimeError');self.assertEqual(closed['context_close_error_kind'],'ValueError')
        self.assertEqual(closed['accepted_phases'],0);self.assertEqual(closed['active_job_cleanup'],'unconfirmed')
        self.assertEqual(closed['unconfirmed_case_write_bytes'],'unknown')
        self.assertNotIn(e.PRIVATE/'pair-result.local.json',files)

    def test_failed_close_receipt_preserves_original_error_and_prevents_successful_return(self):
        for failure,expected in ((True,RuntimeError),(False,OSError)):
            result,error,files,calls=self.simulate(failure=failure,closed_write_failure=True)
            self.assertIsNone(result);self.assertIsInstance(error,expected)
            self.assertNotIn(e.PRIVATE/'closed.local.json',files)
            self.assertEqual(calls,[0] if failure else list(range(56)))


if __name__=='__main__':unittest.main()
