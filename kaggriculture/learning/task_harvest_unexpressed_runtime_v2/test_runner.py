"""Metadata inheritance and isolated CLI regressions; no real private reads."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import contract, runner


def isolated(*args):
    return subprocess.run([sys.executable,'-I','-B',*args],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',timeout=45,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))


def fake_archive(*,accepted=0):
    """A fully synthetic 18-object witness, kept entirely in memory."""
    raw = lambda value: runner.ordered(value)+b'\n'
    objects = {key:raw({'fixture':key}) for key in runner.failure_paths()}
    selected = []
    for source in contract.sources():
        target = dict(schema_version='task-harvest-unexpressed-target-v1',side=source['side'],
            step=240,ordinal=0,unit=1,target=[2,1],input_sha256='a'*64,
            original_unit_request=['WATER'],unit_requests_sha256='b'*64)
        value = dict(source=source,selection=dict(schema_version='task-harvest-unexpressed-selection-v1',
            target=target,eligibility={'fixture':True},counts={'trace_lines':719},reason=None))
        selected.append(value)
        objects['private/'+source['role']+'.selection.local.json'] = raw(value)
    default = b'synthetic default'
    identities = {'synthetic_completed_receipts':'identity'}
    binding = dict(schema='task-harvest-unexpressed-binding-v1',base_commit='c'*40,salt='1a'*32,
        selection_started_sha256=runner.sha(objects['private/selection-started.local.json']),
        completed_sources=identities,default_sha256=runner.sha(default),selections=selected)
    binding_raw = raw(binding)
    digest = runner.sha(binding_raw)
    objects['private/binding.local.json'] = objects['stage/binding.local.json'] = binding_raw
    objects['private/selection-closed.local.json'] = raw(dict(
        schema='task-harvest-unexpressed-selection-closed-v1',outcome='prepared',error_kind=None,
        binding_sha256=digest,source_roles_scanned=4,model_forwards=0,official_transitions=0,
        selection_started_sha256=binding['selection_started_sha256']))
    objects['private/closed.local.json'] = raw(dict(schema='task-harvest-unexpressed-closed-v1',
        registration_commit=runner.PRIOR_COMMIT,binding_sha256=digest,outcome='failed',
        accepted_cases=accepted,result_sha256=None,retry_allowed=False,
        owner_sha256=runner.sha(objects['private/owner.local.json']),
        attempt_sha256=runner.sha(objects['private/attempt.local.json'])))
    objects['stage/F01/case.local.json'] = raw(dict(passed=False,status='failed',retry_allowed=False))
    objects['stage/F01/events.jsonl.gz'] = b'synthetic bytes; never decompressed'
    files = {key:dict(bytes=len(value),sha256=runner.sha(value)) for key,value in objects.items()}
    commitment = runner.sha(bytes.fromhex(binding['salt'])+b'\0task-harvest-unexpressed-failure-v1\0'+
        json.dumps(files,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode())
    witness = dict(schema='task-harvest-unexpressed-failure-witness-v1',registration_commit=runner.PRIOR_COMMIT,
        files=files,salted_commitment_sha256=commitment)
    data = {path:objects[key] for key,path in runner.failure_paths().items()}
    data[runner.PRIOR_WITNESS] = raw(witness)
    data[runner.REPO/runner.PRIOR_RESULTS/'certificate.json'] = raw(dict(
        registration_commit=runner.PRIOR_COMMIT,experiment_passed=False,
        originals=dict(files=18,bytes=sum(map(len,objects.values())),salted_commitment_sha256=commitment)))
    data[runner.REPO/runner.PRIOR_REG] = raw(dict(base_commit=binding['base_commit'],binding_commitment_sha256=digest))
    data[runner.GAME/'main.py'] = default
    calls = []
    def read(path,maximum=16*1024**2):
        calls.append(Path(path))
        value = data[Path(path)]
        if len(value) > maximum:
            raise ValueError('Synthetic bounded read exceeded')
        return value
    return SimpleNamespace(read=read,decode=json.loads),data,calls,commitment,binding


class RunnerTests(unittest.TestCase):
    def test_actual_isolated_help_and_snapshot_metadata_without_private_reads(self):
        result = isolated(str(Path(runner.__file__).resolve()),'--help')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--registration-commit',result.stdout)
        code = r'''
import hashlib, importlib.util, json, sys
from pathlib import Path
def audit(event,args):
    if event == 'import' and any(args[0] == p or args[0].startswith(p+'.') for p in ('torch','kaggle_environments')):
        raise AssertionError('Execution import in metadata preparation')
    if event == 'open' and isinstance(args[0],(str,bytes)):
        text=args[0].decode() if isinstance(args[0],bytes) else args[0]
        path=Path(text)
        if set(part.lower() for part in path.parts)&{'private','artifacts'} or path.suffix.lower() in {'.pt','.pth','.npz','.gz','.jsonl'}:
            raise AssertionError('Private or model read in import-only test')
sys.addaudithook(audit)
entry=Path(sys.argv[1]); spec=importlib.util.spec_from_file_location('_runtime_v2_cli_test',entry)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
shared=m._shared();infra=shared.infrastructure()
relative=m.PREFIX+'contract.py';raw=(m.REPO/relative).read_bytes()
c=infra.snapshot(relative,hashlib.sha256(raw).hexdigest())
assert c.specification()['target_reselection'] is False and c.LIMITS['max_official_transitions']==384
assert len(c.sources())==4
modules=m.metadata_modules();m.no_execution_imports()
assert set(modules)=={'sources','driver'}
print(json.dumps({'snapshot':True,'metadata_only':True,'selector_entry_exposed':False}))
'''
        result = isolated('-c',code,str(Path(runner.__file__).resolve()))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),dict(snapshot=True,metadata_only=True,selector_entry_exposed=False))

    def test_owned_coordinator_isolated_with_new_paths_markers_and_idempotent_schema(self):
        from learning.task_harvest_unexpressed_v1 import runner as historical
        previous = (historical.BASE,historical.PRIVATE,historical.marker,historical._shared)
        one,two = runner._shared(),runner._shared()
        self.assertEqual(Path(one.run.__globals__['__file__']).resolve(),Path(runner.__file__).resolve())
        self.assertEqual(one.PRIVATE,runner.PRIVATE)
        self.assertEqual(one.STAGE,runner.STAGE)
        permit = one.marker('permit','a'*40,'b'*64,'F01')
        self.assertEqual(permit['schema'],runner.DOC+'marker')
        one.PRIVATE = Path('synthetic changed namespace')
        self.assertEqual(two.PRIVATE,runner.PRIVATE)
        self.assertEqual(previous,(historical.BASE,historical.PRIVATE,historical.marker,historical._shared))
        for schema in ('task-harvest-window-accepted-v1','task-harvest-unexpressed-accepted-v1'):
            old = {'schema':schema,'passed':True}
            value = runner.schema_document(old)
            self.assertEqual(value,{'schema':runner.DOC+'accepted','passed':True})
            self.assertEqual(old['schema'],schema)
            self.assertIs(runner.schema_document(value),value)
        current = {'schema':runner.DOC+'selection-started'}
        self.assertIs(runner.schema_document(current),current)

    def test_prepare_payload_only_copies_ordered_points_and_authenticates_receipts_assets(self):
        infra,data,reads,commitment,binding = fake_archive()
        original = deepcopy(data)
        requests,assets = [],[]
        def load(*,guard,verify_game_files):
            guard();requests.append(verify_game_files)
            return SimpleNamespace(identities=binding['completed_sources'],manifests=dict(C01='one',C02='two'))
        def forbidden(*args,**kwargs):
            self.fail('Preparation must not select, read traces or decode Replay')
        driver = SimpleNamespace(_trace=forbidden,_replay=forbidden)
        modules = dict(sources=SimpleNamespace(load=load),driver=driver,
                       selection=SimpleNamespace(select_target=forbidden))
        def validate(manifest,actual_driver):
            self.assertIs(actual_driver,driver);assets.append(manifest);return True
        with patch.object(runner,'FAILURE_COMMITMENT',commitment):
            previous,identities,selected = runner.prepare_payload(infra,modules,
                SimpleNamespace(validate_assets=validate),contract,lambda: True)
        self.assertEqual(requests,[False])
        self.assertEqual(assets,['one','two'])
        self.assertEqual(previous['files'],18)
        self.assertEqual(identities,binding['completed_sources'])
        self.assertEqual(runner.ordered(selected),runner.ordered(binding['selections']))
        self.assertEqual(data,original)
        self.assertEqual(set(reads),set(data))
        selected[0]['selection']['target']['original_unit_request'].append('change isolated copy')
        self.assertEqual(binding['selections'][0]['selection']['target']['original_unit_request'],['WATER'])

    def test_failed_archive_mismatch_or_previous_acceptance_prevents_reproduction(self):
        infra,data,reads,commitment,binding = fake_archive(accepted=1)
        with patch.object(runner,'FAILURE_COMMITMENT',commitment):
            with self.assertRaises(ValueError):
                runner.failed_source(infra,lambda: True)
        infra,data,reads,commitment,binding = fake_archive()
        data[runner.PRIOR_STAGE/'F01/events.jsonl.gz'] += b'changed'
        with patch.object(runner,'FAILURE_COMMITMENT',commitment):
            with self.assertRaises(ValueError):
                runner.failed_source(infra,lambda: True)

    def test_point_order_must_match_original_bytes_and_old_budget_gate_is_unchanged(self):
        infra,data,reads,commitment,binding = fake_archive()
        with patch.object(runner,'FAILURE_COMMITMENT',commitment):
            previous = runner.failed_source(infra,lambda: True)
        role = 'private/F01.selection.local.json'
        value = json.loads(previous['originals'][role])
        value['source'] = dict(reversed(list(value['source'].items())))
        previous['originals'][role] = runner.ordered(value)+b'\n'
        with self.assertRaises(ValueError):
            runner.inherited_points(previous,contract)
        from learning.task_harvest_unexpressed_v1 import contract as old
        self.assertEqual(contract.LIMITS,old.LIMITS)
        self.assertEqual(contract.sources(),old.sources())


if __name__ == '__main__':
    unittest.main()
