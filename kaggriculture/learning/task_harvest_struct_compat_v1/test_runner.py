"""Entry, frozen-point and provenance regressions using only synthetic bytes."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import contract,runner


def fake_previous():
    selected = []
    for source in contract.sources():
        target = dict(schema_version='task-harvest-unexpressed-target-v1',step=240,
            side=source['side'],ordinal=0,unit=0,target=[1,1],input_sha256='a'*64,
            original_unit_request=['WATER'],unit_requests_sha256='b'*64)
        selected.append(dict(source=source,selection=dict(schema_version='task-harvest-unexpressed-selection-v1',
            target=target,counts=dict(trace_lines=719),reason=None,eligibility={})))
    default = b'unchanged synthetic default'
    binding = dict(completed_sources={'synthetic_receipts':32},default_sha256=runner.sha(default),selections=selected)
    originals = {'private/'+v['source']['role']+'.selection.local.json':runner.ordered(v)+b'\n' for v in selected}
    return dict(binding=binding,originals=originals,binding_sha256='c'*64,
        closed_sha256='d'*64,witness_sha256='e'*64,salted_commitment_sha256=runner.FAILURE_COMMITMENT,
        files=18,bytes=187746),default


def fake_diagnostic(previous,*,passed=False,outcome='located_original_error',prior_binding=None):
    raw = lambda v:runner.ordered(v)+b'\n'
    files = {key:raw({'synthetic':key}) for key in runner.diagnostic_paths()}
    files['private/F01.selection.local.json'] = previous['originals']['private/F01.selection.local.json']
    binding = dict(schema=runner.DIAGNOSTIC_DOC+'binding',base_commit='f'*40,salt='ac'*32,
        prior_binding_sha256=previous['binding_sha256'] if prior_binding is None else prior_binding,
        completed_sources=previous['binding']['completed_sources'],default_sha256=previous['binding']['default_sha256'],
        selections=deepcopy(previous['binding']['selections'][:1]),
        selection_started_sha256=runner.sha(files['private/selection-started.local.json']))
    binding_raw = raw(binding);binding_sha = runner.sha(binding_raw)
    files['private/binding.local.json'] = files['stage/binding.local.json'] = binding_raw
    files['private/selection-closed.local.json'] = raw(dict(schema=runner.DIAGNOSTIC_DOC+'selection-closed',
        outcome='prepared',binding_sha256=binding_sha,source_roles_copied=1,trace_rows_read=0,
        target_reselection=False,model_forwards=0,official_transitions=0,replay_decodes=0,
        selection_started_sha256=binding['selection_started_sha256']))
    meter = dict(calls=dict(arm_attempts=1,restore_returns=1,official_step_attempts=1,official_step_returns=1,
                           actor_returns=241,rival_returns=241))
    files['stage/F01/case.local.json'] = raw(dict(schema='task-harvest-unexpressed-case-v1',
        passed=passed,status='failed',meter=meter))
    diagnostic = dict(passed=True,outcome=outcome,error_id='STATIC_eb5d37bb2b6c6eaf48d5',
        location_kind='direct_raise',rejected_value_kind='official_Struct',comparison=dict(
            semantic_frame_equal=True,ordered_frame_equal=True,
            original_type_request_check=dict(outcome='raises',error_kind='ValueError')))
    files['stage/F01/accepted.local.json'] = raw(dict(diagnostic_passed=True,original_case_passed=False,
        case_receipt=dict(sha256=runner.sha(files['stage/F01/case.local.json'])),diagnostic=diagnostic))
    accepted = [dict(role='F01',sha256=runner.sha(files['stage/F01/accepted.local.json']))]
    files['private/result.local.json'] = raw(dict(original_case_passed=False,accepted_pairs=0,
        full_horizon_verified=False,economic_value_evaluated=False,accepted=accepted,diagnostic=diagnostic,meter=meter))
    result_sha = runner.sha(files['private/result.local.json'])
    files['private/closed.local.json'] = raw(dict(schema=runner.DIAGNOSTIC_DOC+'closed',
        registration_commit=runner.DIAGNOSTIC_COMMIT,binding_sha256=binding_sha,outcome='diagnostic_complete',
        error_kind=None,accepted_diagnostics=1,accepted_pairs=0,retry_allowed=False,
        owner_sha256=runner.sha(files['private/owner.local.json']),attempt_sha256=runner.sha(files['private/attempt.local.json']),
        result_sha256=result_sha))
    files['private/certificate.local.json'] = raw(dict(closed_sha256=runner.sha(files['private/closed.local.json']),
        result_sha256=result_sha,diagnostic_passed=True,original_case_passed=False,accepted=accepted,diagnostic=diagnostic))
    other = sum(len(v) for k,v in files.items() if k != 'stage/F01/events.jsonl.gz')
    files['stage/F01/events.jsonl.gz'] = b'opaque old event bytes'+b'x'*(246642-other-22)
    # Padding is opaque; metadata preparation hashes it and never decompresses it.
    assert sum(map(len,files.values())) == 246642
    inventory = {k:dict(bytes=len(v),sha256=runner.sha(v)) for k,v in files.items()}
    commitment = runner.sha(bytes.fromhex(binding['salt'])+b'\0task-harvest-first-step-diagnostic-v1\0'+
        json.dumps(inventory,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode())
    witness = dict(schema='task-harvest-first-step-diagnostic-witness-v1',registration_commit=runner.DIAGNOSTIC_COMMIT,
        files=inventory,salted_commitment_sha256=commitment)
    data = {path:files[key] for key,path in runner.diagnostic_paths().items()}
    data[runner.DIAGNOSTIC_WITNESS] = raw(witness)
    data[runner.REPO/runner.DIAGNOSTIC_REG] = raw(dict(base_commit=binding['base_commit'],binding_commitment_sha256=binding_sha))
    data[runner.REPO/runner.DIAGNOSTIC_RESULTS/'certificate.json'] = raw(dict(registration_commit=runner.DIAGNOSTIC_COMMIT,
        diagnostic_collection_passed=True,original_case_passed=False,official_steps_returned=1,accepted_pairs=0,
        originals=dict(files=18,bytes=246642,salted_commitment_sha256=commitment)))
    reads = []
    def read(path,maximum=16*1024**2):
        path=Path(path);reads.append(path);value=data[path]
        if len(value)>maximum:raise ValueError('Synthetic bounded read')
        return value
    return SimpleNamespace(read=read,decode=json.loads),data,reads,commitment


class RunnerTests(unittest.TestCase):
    def test_actual_isolated_help_and_metadata_snapshot(self):
        def isolated(*args):
            return subprocess.run([sys.executable,'-I','-B',*args],stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',timeout=45,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        path=str(Path(runner.__file__).resolve())
        result=isolated(path,'--help')
        self.assertEqual(result.returncode,0,result.stderr)
        code=r'''
import hashlib,importlib.util,json,sys
from pathlib import Path
def audit(event,args):
    if event=='import' and any(args[0]==p or args[0].startswith(p+'.') for p in ('torch','kaggle_environments')):
        raise AssertionError('Execution import during metadata check')
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0])
        if set(x.lower() for x in p.parts)&{'private','artifacts'} or p.suffix.lower() in {'.gz','.jsonl','.pt','.pth','.npz'}:
            raise AssertionError('Private/model read during metadata check')
sys.addaudithook(audit)
spec=importlib.util.spec_from_file_location('_compat_entry_test',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
legacy=m._legacy();shared=m._shared(legacy);infra=shared.infrastructure()
p=m.PREFIX+'contract.py';raw=(m.REPO/p).read_bytes();c=infra.snapshot(p,hashlib.sha256(raw).hexdigest())
assert c.LIMITS['max_official_transitions']==384 and len(c.sources())==4
assert set(legacy.metadata_modules())=={'sources','driver'}
assert Path(shared.run.__globals__['__file__']).resolve()==Path(sys.argv[1]).resolve()
assert shared.PRIVATE==m.PRIVATE and shared.STAGE==m.STAGE
legacy.no_execution_imports();print(json.dumps({'metadata_only':True,'roles':4}))
'''
        result=isolated('-c',code,path)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),dict(metadata_only=True,roles=4))

    def test_exact_diagnostic_inventory_requires_failed_original_and_linked_origin(self):
        previous,_=fake_previous()
        infra,data,reads,commitment=fake_diagnostic(previous)
        before=deepcopy(data)
        with patch.object(runner,'DIAGNOSTIC_COMMITMENT',commitment):
            value=runner.diagnostic_source(infra,lambda:True,previous)
        self.assertEqual((value['files'],value['bytes']),(18,246642))
        self.assertEqual(set(reads),set(data));self.assertEqual(data,before)
        for kw in (dict(passed=True),dict(outcome='first_step_complete'),dict(prior_binding='9'*64)):
            infra,data,_,commitment=fake_diagnostic(previous,**kw)
            with patch.object(runner,'DIAGNOSTIC_COMMITMENT',commitment):
                with self.assertRaises(ValueError):runner.diagnostic_source(infra,lambda:True,previous)
        infra,data,_,commitment=fake_diagnostic(previous)
        data[runner.DIAGNOSTIC_STAGE/'F01/events.jsonl.gz']+=b'tampered'
        with patch.object(runner,'DIAGNOSTIC_COMMITMENT',commitment):
            with self.assertRaises(ValueError):runner.diagnostic_source(infra,lambda:True,previous)

    def test_prepare_copies_four_points_without_selection_or_game_reads(self):
        previous,default=fake_previous()
        infra,data,reads,commitment=fake_diagnostic(previous)
        data[runner.GAME/'main.py']=default
        original=deepcopy(previous);loads=[];assets=[]
        def forbidden(*a,**kw):self.fail('Preparation selected a point or decoded a source game')
        def load(*,guard,verify_game_files):
            guard();loads.append(verify_game_files)
            return SimpleNamespace(identities=previous['binding']['completed_sources'],manifests=dict(C01=1,C02=2))
        driver=SimpleNamespace(_trace=forbidden,_replay=forbidden)
        modules=dict(sources=SimpleNamespace(load=load),driver=driver,selection=SimpleNamespace(select_target=forbidden))
        def validate(manifest,dm):self.assertIs(dm,driver);assets.append(manifest);return True
        with patch.object(runner,'failed_source',return_value=previous),\
                patch.object(runner,'DIAGNOSTIC_COMMITMENT',commitment),\
                patch.object(runner,'utils_identity',return_value=contract.STRUCT_UTILS_SHA256):
            _,diagnostic,identities,points=runner.prepare_payload(infra,modules,
                SimpleNamespace(validate_assets=validate),contract,lambda:True)
        self.assertEqual(loads,[False]);self.assertEqual(assets,[1,2])
        self.assertEqual(identities,original['binding']['completed_sources'])
        self.assertEqual(runner.ordered(points),runner.ordered(original['binding']['selections']))
        self.assertEqual(previous,original);self.assertEqual(diagnostic['files'],18)
        self.assertEqual(set(reads),set(data))
        points[0]['selection']['target']['original_unit_request'].append('isolated')
        self.assertEqual(previous,original)
        broken=deepcopy(previous)
        case=json.loads(broken['originals']['private/F01.selection.local.json'])
        case['source']=dict(reversed(list(case['source'].items())))
        broken['originals']['private/F01.selection.local.json']=runner.ordered(case)+b'\n'
        with self.assertRaises(ValueError):runner._legacy().inherited_points(broken,contract)

    def test_frozen_coordinator_isolated_and_budget_gate_unchanged(self):
        from learning.task_harvest_unexpressed_runtime_v2 import runner as old_runner
        from learning.task_harvest_unexpressed_runtime_v2 import contract as old_contract
        previous=(old_runner.BASE,old_runner.PRIVATE,old_runner._shared,old_runner.binding_read)
        one,two=runner._shared(),runner._shared()
        one.PRIVATE=Path('synthetic independent namespace')
        self.assertEqual(two.PRIVATE,runner.PRIVATE)
        self.assertEqual(previous,(old_runner.BASE,old_runner.PRIVATE,old_runner._shared,old_runner.binding_read))
        self.assertEqual(contract.LIMITS,old_contract.LIMITS)
        self.assertEqual(contract.sources(),old_contract.sources())
        for role in ('F01','F02','F03','F04'):
            self.assertEqual(two.marker('permit','a'*40,'b'*64,role)['schema'],runner.DOC+'marker')
        from learning.task_harvest_unexpressed_v1 import contract as base_contract
        self.assertEqual(base_contract.HORIZON,contract.HORIZON)
        original=ValueError('synthetic original');secondary=OSError('synthetic closure')
        def fail(value):raise secondary
        self.assertFalse(two.close_preserving_primary(fail,{},original))
        with self.assertRaises(OSError) as raised:two.close_preserving_primary(fail,{},None)
        self.assertIs(raised.exception,secondary)

    def test_official_utils_identity_rejects_different_bytes_without_importing(self):
        package=Path('synthetic official package')
        raw=b'synthetic authenticated utils'
        spec=SimpleNamespace(submodule_search_locations=[str(package)])
        expected=SimpleNamespace(STRUCT_UTILS_SHA256=runner.sha(raw))
        reads=[]
        def read(path):reads.append(path);return raw
        with patch.object(runner.importlib.util,'find_spec',return_value=spec):
            self.assertEqual(runner.utils_identity(SimpleNamespace(read=read),expected),runner.sha(raw))
            with self.assertRaises(ValueError):
                runner.utils_identity(SimpleNamespace(read=lambda p:raw+b'changed'),expected)
        self.assertEqual(reads,[package/'utils.py'])


if __name__ == '__main__':
    unittest.main()
