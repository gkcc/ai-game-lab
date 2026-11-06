"""First-step entry contracts with synthetic in-memory records only."""
from contextlib import nullcontext,redirect_stdout,redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import contract,runner


def checked(outcome='located_original_error'):
    return dict(document=dict(passed=False,status='failed',meter={}),
        receipt={'path':'case.local.json','sha256':'a'*64,'bytes':123},
        diagnostic=dict(passed=True,outcome=outcome))


def fake_archive():
    raw = lambda value:runner.ordered(value)+b'\n'
    files = {key:raw({'synthetic':key}) for key in runner.failure_paths()}
    selected = []
    for i,role in enumerate(('F01','F02','F03','F04')):
        source = deepcopy(contract.sources()[0]);source['role'] = role
        target = dict(schema_version='task-harvest-unexpressed-target-v1',step=240,
            side=0,ordinal=0,unit=i,target=[1,1],input_sha256='a'*64,
            original_unit_request=['WATER'],unit_requests_sha256='b'*64)
        value = dict(source=source,selection=dict(schema_version='task-harvest-unexpressed-selection-v1',
            target=target,counts=dict(trace_lines=719),reason=None,eligibility={}))
        selected.append(value);files['private/'+role+'.selection.local.json'] = raw(value)
    default = b'synthetic unchanged default'
    binding = dict(schema=runner.PRIOR_DOC+'binding',base_commit='c'*40,salt='ab'*32,
        selection_started_sha256=runner.sha(files['private/selection-started.local.json']),
        completed_sources={'synthetic_receipts':'32'},default_sha256=runner.sha(default),selections=selected)
    binding_raw = raw(binding);binding_sha = runner.sha(binding_raw)
    files['private/binding.local.json'] = files['stage/binding.local.json'] = binding_raw
    files['private/closed.local.json'] = raw(dict(schema=runner.PRIOR_DOC+'closed',
        registration_commit=runner.PRIOR_COMMIT,binding_sha256=binding_sha,outcome='failed',
        accepted_cases=0,result_sha256=None,retry_allowed=False,
        owner_sha256=runner.sha(files['private/owner.local.json']),
        attempt_sha256=runner.sha(files['private/attempt.local.json'])))
    files['private/selection-closed.local.json'] = raw(dict(schema=runner.PRIOR_DOC+'selection-closed',
        outcome='prepared',error_kind=None,source_roles_copied=4,binding_sha256=binding_sha,
        trace_rows_read=0,target_reselection=False,model_forwards=0,official_transitions=0,replay_decodes=0,
        selection_started_sha256=binding['selection_started_sha256']))
    files['stage/F01/case.local.json'] = raw(dict(schema='task-harvest-unexpressed-case-v1',
        passed=False,status='failed',error_kind='invalid',meter=dict(calls=dict(
            arm_attempts=1,restore_returns=1,official_step_attempts=1,official_step_returns=1,
            actor_returns=241,rival_returns=241)),runtime_preparation=dict(
                status='prepared',run_pair_attempts=1,run_pair_returns=0)))
    # Opaque failed-event bytes are hashed but never decoded by preparation.
    others = sum(len(value) for key,value in files.items() if key != 'stage/F01/events.jsonl.gz')
    files['stage/F01/events.jsonl.gz'] = b'x'*(187746-others)
    inventory = {key:dict(bytes=len(value),sha256=runner.sha(value)) for key,value in files.items()}
    commitment = runner.sha(bytes.fromhex(binding['salt'])+b'\0task-harvest-unexpressed-runtime-failure-v2\0'+
        json.dumps(inventory,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode())
    witness = dict(schema='task-harvest-unexpressed-runtime-failure-witness-v2',registration_commit=runner.PRIOR_COMMIT,
                   files=inventory,salted_commitment_sha256=commitment)
    data = {path:files[key] for key,path in runner.failure_paths().items()}
    data[runner.PRIOR_WITNESS] = raw(witness)
    data[runner.REPO/runner.PRIOR_REG] = raw(dict(base_commit=binding['base_commit'],binding_commitment_sha256=binding_sha))
    data[runner.REPO/runner.PRIOR_RESULTS/'certificate.json'] = raw(dict(registration_commit=runner.PRIOR_COMMIT,
        experiment_passed=False,originals=dict(files=18,bytes=187746,salted_commitment_sha256=commitment)))
    data[runner.GAME/'main.py'] = default
    reads = []
    def read(path,maximum=16*1024**2):
        reads.append(Path(path));value = data[Path(path)]
        if len(value)>maximum:
            raise ValueError('Synthetic bounded read exceeded')
        return value
    return SimpleNamespace(read=read,decode=json.loads),data,reads,commitment,binding


class RunnerTests(unittest.TestCase):
    def test_actual_isolated_help_and_stdlib_snapshot_metadata(self):
        def isolated(*args):
            return subprocess.run([sys.executable,'-I','-B',*args],stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',timeout=45,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        result = isolated(str(Path(runner.__file__).resolve()),'--help')
        self.assertEqual(result.returncode,0,result.stderr)
        code = r'''
import hashlib,importlib.util,json,sys
from pathlib import Path
def audit(event,args):
    if event=='import' and any(args[0]==p or args[0].startswith(p+'.') for p in ('torch','kaggle_environments')):
        raise AssertionError('Execution import in metadata probe')
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0])
        if set(x.lower() for x in p.parts)&{'private','artifacts'} or p.suffix.lower() in {'.gz','.jsonl','.pt','.pth','.npz'}:
            raise AssertionError('Private/model read in metadata probe')
sys.addaudithook(audit)
spec=importlib.util.spec_from_file_location('_diagnostic_entry_test',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
legacy=m._legacy();shared=m._shared(legacy);infra=shared.infrastructure()
path=m.PREFIX+'contract.py';raw=(m.REPO/path).read_bytes();c=infra.snapshot(path,hashlib.sha256(raw).hexdigest())
assert c.LIMITS['max_official_transitions']==1 and c.LIMITS['max_controller_calls']==241
assert [s['role'] for s in c.sources()]==['F01']
assert set(legacy.metadata_modules())=={'sources','driver'}
assert Path(shared.run.__globals__['__file__']).resolve()==Path(sys.argv[1]).resolve()
legacy.no_execution_imports();print(json.dumps({'metadata_only':True,'source_roles':1}))
'''
        result = isolated('-c',code,str(Path(runner.__file__).resolve()))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),dict(metadata_only=True,source_roles=1))

    def test_prepare_inherits_only_F01_without_reselection_or_trace_reads(self):
        infra,data,reads,commitment,binding = fake_archive()
        legacy = runner._legacy();requests=[]
        def forbidden(*args,**kwargs):self.fail('No selection or historical decoding during preparation')
        def load(*,guard,verify_game_files):
            guard();requests.append(verify_game_files)
            return SimpleNamespace(identities=binding['completed_sources'],manifests=dict(C01=1,C02=2))
        modules=dict(sources=SimpleNamespace(load=load),driver=SimpleNamespace(_trace=forbidden,_replay=forbidden),
                     selection=SimpleNamespace(select_target=forbidden))
        before=deepcopy(data)
        with patch.object(runner,'FAILURE_COMMITMENT',commitment):
            previous,identities,cases = legacy.prepare_payload(infra,modules,
                SimpleNamespace(validate_assets=lambda m,d:True),contract,lambda:True)
        self.assertEqual(requests,[False]);self.assertEqual(previous['bytes'],187746)
        self.assertEqual([c['source']['role'] for c in cases],['F01'])
        self.assertEqual(runner.ordered(cases[0]),runner.ordered(binding['selections'][0]))
        self.assertEqual(set(reads),set(data));self.assertEqual(data,before)
        previous['binding']['selections'][0]['selection']['target']['step']=241
        with self.assertRaises(ValueError):runner.inherited_points(previous,contract)

    def test_original_failure_remains_false_while_diagnosis_can_be_accepted(self):
        class Complete(Exception):pass
        for error,outcome in ((ValueError('synthetic'), 'located_original_error'),(Complete(),'first_step_complete')):
            def execute(*args):raise error
            worker=SimpleNamespace(execute_case=execute,verify_case=lambda *args:checked(outcome))
            value=runner.execute_and_verify(worker,{},None,dict(source={},selection={}),None,lambda:True,contract)
            self.assertFalse(value['document']['passed']);self.assertEqual(value['diagnostic']['outcome'],outcome)
        for error in (MemoryError(),TimeoutError(),OSError(),SystemExit(0)):
            def execute(*args):raise error
            worker=SimpleNamespace(execute_case=execute,verify_case=lambda *args:self.fail('Resource/exit errors cannot be accepted'))
            with self.assertRaises(type(error)):
                runner.execute_and_verify(worker,{},None,dict(source={},selection={}),None,lambda:True,contract)
        worker=SimpleNamespace(execute_case=lambda *args:None,verify_case=lambda *args:checked())
        with self.assertRaises(ValueError):runner.execute_and_verify(worker,{},None,dict(source={},selection={}),None,lambda:True,contract)
        invalid=checked();invalid['document']['passed']=True
        with self.assertRaises(ValueError):contract.accept_diagnostic(invalid)

    def test_secondary_validation_error_preserves_the_original_exception_object(self):
        for stage in ('check','missing_receipt','contract'):
            with self.subTest(stage=stage):
                original=ValueError('synthetic original execution failure')
                secondary=FileNotFoundError('synthetic missing diagnostic receipt')
                def execute(*args):raise original
                def fail(*args):raise secondary
                worker=SimpleNamespace(execute_case=execute,
                    verify_case=fail if stage=='missing_receipt' else lambda *args:checked())
                gate=fail if stage=='check' else lambda:True
                validator=SimpleNamespace(accept_diagnostic=fail) if stage=='contract' else contract
                with self.assertRaises(ValueError) as raised:
                    runner.execute_and_verify(worker,{},None,dict(source={},selection={}),None,gate,validator)
                self.assertIs(raised.exception,original)
                self.assertIs(raised.exception.__cause__,secondary)

    def test_real_child_dispatch_keeps_existing_stage_within_single_total_cap(self):
        """Run the production child branch against memory-only source/receipt IO."""
        commit='a'*40;source=contract.sources()[0]
        case=dict(source=source,selection=dict(target=dict(step=240)))
        binding=dict(selections=[case],default_sha256=runner.sha(b'default'),completed_sources={})
        raw=runner.ordered(binding)+b'\n';digest=runner.sha(raw)
        owner=dict(deadline=time.monotonic()+1000)
        data={runner.PRIVATE/'binding.local.json':raw,runner.STAGE/'binding.local.json':raw,
            runner.PRIVATE/'owner.local.json':runner.ordered(owner),runner.GAME/'main.py':b'default',
            runner.PRIVATE/'attempt.local.json':runner.ordered(runner.marker('attempt',commit,digest)),
            runner.STAGE/'F01/permit.local.json':runner.ordered(runner.marker('permit',commit,digest,'F01'))}
        def read(path,*args):return data[Path(path)]
        def write(path,value,*args):
            self.assertNotIn(Path(path),data);data[Path(path)]=runner.ordered(value);return runner.sha(data[Path(path)])
        infra=SimpleNamespace(read=read,write=write,decode=json.loads,snapshot=lambda *args:None)
        identity=SimpleNamespace(verify_public=lambda **kwargs:None)
        base=SimpleNamespace(physical_available=lambda:10*1024**3)
        executed=[]
        def execute(modules,context,s,selection,out,check):
            check();executed.append((s['role'],out));raise ValueError('synthetic located error')
        actual_worker=SimpleNamespace(execute_case=execute,verify_case=lambda *args:checked())
        modules=dict(driver=SimpleNamespace(),window_worker=actual_worker,
            worker=SimpleNamespace(_quiet_output=nullcontext),fresh=SimpleNamespace(verify_live_owner=lambda *args,**kwargs:None))
        reserves=[]
        def budget(dm,c,reserve=0):
            reserves.append(reserve)
            # Existing binding/permit/readiness means a second 128 MiB reserve
            # cannot fit; only the already-owned child budget may be checked.
            if 1024+reserve>c.LIMITS['total_output_bytes']:
                raise ValueError('Existing stage plus redundant reserve exceeds total')
        shared=SimpleNamespace(READY='ready',MAX_SMALL=1024**2,
            authenticate=lambda value:(infra,{'public_sources':{'ready':'b'*64}},contract,identity,None,base),
            runtime=lambda *args:modules,source_context=lambda *args,**kwargs:SimpleNamespace(identities={}),
            budget_check=budget)
        with patch.object(runner,'_shared',return_value=shared),patch.object(runner,'binding_read',return_value=(raw,binding)),\
                patch.object(runner.os.path,'lexists',return_value=False):
            self.assertIsNone(runner.run(commit,role='F01'))
        self.assertEqual(executed,[('F01',runner.STAGE/'F01')])
        self.assertEqual(reserves,[0])
        self.assertEqual(json.loads(data[runner.STAGE/'F01/consumed.local.json'])['role'],'F01')
        self.assertNotIn(runner.PRIVATE/'closed.local.json',data)

    def test_other_role_rejected_before_authentication_and_no_old_global_mutation(self):
        from learning.task_harvest_unexpressed_runtime_v2 import runner as old
        values=(old.BASE,old.PRIVATE,old._shared,old.marker)
        one,two=runner._shared(),runner._shared()
        one.PRIVATE=Path('synthetic namespace only')
        self.assertEqual(two.PRIVATE,runner.PRIVATE)
        self.assertEqual(values,(old.BASE,old.PRIVATE,old._shared,old.marker))
        with patch.object(runner,'_shared',side_effect=AssertionError('Must not authenticate another role')):
            with self.assertRaises(ValueError):runner.run('a'*40,role='F02')
        with self.assertRaises(ValueError):runner.marker('permit','a'*40,'b'*64,'F02')


if __name__ == '__main__':
    unittest.main()
