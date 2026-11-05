"""Authenticate the closed F screening run, never an unused seed inventory.

Imports perform no private I/O. Historical public sources are checked against
their registration Git blob, rather than pretending the current HEAD is that
old registration. Only PROGRESS may have an appended suffix. The publication
witness authenticates exactly the original 63 files; it is not a run output.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

REPO = Path(__file__).resolve().parents[3]
GAME = REPO.parent.parent/'kaggriculture'
BASE = '5a0dd353034889940d4215def53b5ddffd3f32c9'
SCREEN_COMMIT = '7c8841019ebb5265ee23aa5e48b7598aaaeb6dc6'
SCREEN_EVIDENCE = 'review/evidence/task-teacher-f-counterfactual-screen-v1-20261106'
SCREEN_RESULTS = 'review/evidence/task-teacher-f-counterfactual-screen-v1-result-20261106'
PRIVATE = REPO.parent/'private/task-teacher-f-counterfactual-screen-v1-20261106'
STAGE = GAME/'artifacts/task-teacher-f-counterfactual-screen-v1-20261106'
PROGRESS = 'review/PROGRESS.md'
DOMAIN = b'\0teacher-f-screen-result-v1\0'
PRIVATE_FILES = ('binding','attempt','owner','screen-result','projected-summary','screen-seal','closed')
SOURCE_FILES = ('readiness','permit','started','consumed','result','supervision','accepted')
FILE_KEYS = frozenset(['private/'+x+'.local.json' for x in PRIVATE_FILES] +
    [f'screen/source-{i:02d}/{x}.local.json' for i in range(8) for x in SOURCE_FILES])
WITNESS = 'publication-witness.local.json'


def require(value):
    if not value: raise ValueError('Completed F screening evidence differs')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def digest(value): return type(value) is str and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def ordered_cases(results, contract):
    """All hits, exact source order; no empty, duplicate or failed-source filter."""
    require(type(results) is list and len(results)==len(contract.SOURCES)==8)
    cases=[]
    for index,result in enumerate(results):
        require(type(result) is dict and contract.source_index(result.get('source'))==index
            and type(result.get('candidate')) is dict)
        case=result['candidate']
        require(contract.source_index(dict(role=case.get('role'),opponent=case.get('opponent'),
            side=case.get('side'),policy='baseline'))==index)
        cases.append(dict(case))
    return cases


def public(infra, git, contract):
    """Stdlib-only historical proof; the callbacks also permit in-memory tests."""
    reg_raw=git('show',SCREEN_COMMIT+':'+SCREEN_EVIDENCE+'/registration.json')
    reg=infra.decode(reg_raw);contract.validate_registration(reg)
    certificate_raw=git('show',BASE+':'+SCREEN_RESULTS+'/certificate.json')
    certificate=infra.decode(certificate_raw)
    summary_raw=git('show',BASE+':'+SCREEN_RESULTS+'/summary.json')
    require(infra.read(REPO/SCREEN_EVIDENCE/'registration.json')==reg_raw
        and infra.read(REPO/SCREEN_RESULTS/'certificate.json')==certificate_raw
        and infra.read(REPO/SCREEN_RESULTS/'summary.json')==summary_raw)
    require(certificate.get('schema')=='teacher-f-screen-result-certificate-v1'
        and certificate.get('registration_commit')==SCREEN_COMMIT
        and certificate.get('registration_sha256')==sha(reg_raw)
        and certificate.get('summary_sha256')==sha(summary_raw)
        and certificate.get('private_runtime_files')==63
        and certificate.get('commitment_domain')=='teacher-f-screen-result-v1'
        and digest(certificate.get('private_runtime_commitment_sha256'))
        and certificate.get('outcome')=='completed' and certificate.get('accepted_sources')==8
        and certificate.get('official_transitions')==16)
    frozen=reg['public_sources']
    require(type(frozen) is dict and len(frozen)==377 and PROGRESS in frozen)
    for relative,expected in frozen.items():
        original=git('show',SCREEN_COMMIT+':'+relative)
        current=infra.read(REPO/relative)
        require(digest(expected) and sha(original)==expected)
        if relative==PROGRESS:
            published=git('show',BASE+':'+relative)
            require(published.startswith(original) and current.startswith(published))
        else:require(current==original)
    return dict(registration=reg,certificate=certificate,summary=infra.decode(summary_raw),
        identity=dict(registration_commit=SCREEN_COMMIT,registration_sha256=sha(reg_raw),
            certificate_sha256=sha(certificate_raw),summary_sha256=sha(summary_raw),
            critical_files_unchanged=376,progress_policy='base_published_bytes_prefix_then_append'))


def commitment(files,salt):
    require(type(salt) is str and re.fullmatch(r'[0-9a-f]{64}',salt) is not None)
    return sha(bytes.fromhex(salt)+DOMAIN+json.dumps(files,sort_keys=True,separators=(',',':')).encode())


def verify_manifest(files,raws,salt,expected):
    require(type(files) is dict and set(files)==FILE_KEYS and set(raws)==FILE_KEYS)
    for key,item in files.items():
        require(type(item) is dict and set(item)=={'sha256','bytes'} and digest(item['sha256'])
            and type(item['bytes']) is int and 0<item['bytes']<=16*1024**2
            and len(raws[key])==item['bytes'] and sha(raws[key])==item['sha256'])
    require(commitment(files,salt)==expected)


def _path(key):
    require(key in FILE_KEYS)
    prefix,relative=key.split('/',1)
    return (PRIVATE if prefix=='private' else STAGE)/relative


def _inventory(root,prefix):
    found=set();count=0
    for folder,dirs,files in os.walk(root,followlinks=False):
        for name in (*dirs,*files):
            path=Path(folder)/name;info=path.lstat();count+=1
            require(count<=80 and not stat.S_ISLNK(info.st_mode)
                and not getattr(info,'st_file_attributes',0)&0x400)
        for name in files: found.add(prefix+'/'+(Path(folder)/name).relative_to(root).as_posix())
    return found


def _marker(kind,binding,index=None):
    value=dict(schema='teacher-f-screen-marker-v1',kind=kind,registration_commit=SCREEN_COMMIT,
        binding_sha256=binding,retry_allowed=False)
    if kind!='attempt':value['source_index']=index
    return value


class Evidence:
    """Private evidence and a bounded F source context; do not print this object."""
    def __init__(self,context,identity,cases,plans,projection):
        self.context,self.identity,self.cases,self.plans,self.projection=context,identity,cases,plans,projection

    def close(self): self.context.close()

    def verify_files(self,infra,guard):
        for key,item in self.identity['files'].items():
            guard();raw=infra.read(_path(key))
            require(len(raw)==item['bytes'] and sha(raw)==item['sha256'])
        require(sha(infra.read(PRIVATE/WITNESS))==self.identity['witness_sha256'])


def load(infra,modules,contract,proof,guard,*,verify_game_files=True):
    """Authenticate 63 receipts and the completed official source chain only."""
    guard()
    require(_inventory(PRIVATE,'private')=={x for x in FILE_KEYS if x.startswith('private/')}|{'private/'+WITNESS}
        and _inventory(STAGE,'screen')=={x for x in FILE_KEYS if x.startswith('screen/')})
    raws={}
    for key in sorted(FILE_KEYS):guard();raws[key]=infra.read(_path(key))
    values={key:infra.decode(raw) for key,raw in raws.items()}
    binding=values['private/binding.local.json'];binding_sha=sha(raws['private/binding.local.json'])
    require(binding.get('schema')=='teacher-f-screen-binding-v1'
        and binding.get('base_commit')==proof['registration']['base_commit']
        and binding_sha==proof['registration']['binding_commitment_sha256'])
    witness_raw=infra.read(PRIVATE/WITNESS);witness=infra.decode(witness_raw)
    require(set(witness)=={'schema','files','commitment_sha256','witness_excluded'}
        and witness['schema']=='teacher-f-screen-publication-witness-v1' and witness['witness_excluded'] is True
        and witness['commitment_sha256']==proof['certificate']['private_runtime_commitment_sha256'])
    verify_manifest(witness['files'],raws,binding['salt'],witness['commitment_sha256'])
    require(values['private/attempt.local.json']==_marker('attempt',binding_sha))
    owner=values['private/owner.local.json']
    require(set(owner)=={'schema','registration_commit','binding_sha256','pid','created_us','deadline'}
        and owner['schema']=='task-pair-fresh-multi-owner-v1' and owner['registration_commit']==SCREEN_COMMIT
        and owner['binding_sha256']==binding_sha and type(owner['pid']) is int and owner['pid']>0
        and type(owner['created_us']) is int and owner['created_us']>0
        and type(owner['deadline']) in (int,float) and math.isfinite(owner['deadline']))
    results=[];accepted=[];jobs=[]
    for index in range(8):
        prefix=f'screen/source-{index:02d}/'
        for kind in ('permit','started','consumed'):
            require(values[prefix+kind+'.local.json']==_marker(kind,binding_sha,index))
        readiness=values[prefix+'readiness.local.json']
        require(readiness.get('consecutive_samples')==3 and type(readiness.get('last_available_bytes')) is int
            and readiness['last_available_bytes']>=contract.LIMITS['between_available_bytes']
            and type(readiness.get('elapsed_seconds')) in (int,float)
            and 20<=readiness['elapsed_seconds']<=contract.LIMITS['readiness_timeout_seconds'])
        job=values[prefix+'supervision.local.json'];job_sha=sha(raws[prefix+'supervision.local.json'])
        require(modules['acceptance'].supervision_ok(job))
        expected=dict(schema='teacher-f-screen-accepted-v1',source_index=index,passed=True,
            registration_commit=SCREEN_COMMIT,binding_sha256=binding_sha,
            result_sha256=sha(raws[prefix+'result.local.json']),supervision_sha256=job_sha,retry_allowed=False)
        require(values[prefix+'accepted.local.json']==expected)
        accepted.append(dict(source_index=index,sha256=sha(raws[prefix+'accepted.local.json'])))
        jobs.append(dict(source_index=index,sha256=job_sha,launcher_reaped=job.get('launcher_reaped'),
            job_members_zero=job.get('job_members_zero'),cleanup_error_kind=job.get('cleanup_error_kind')))
        results.append(values[prefix+'result.local.json'])
    require(values['private/screen-result.local.json']==dict(schema='teacher-f-screen-result-v1',
        registration_commit=SCREEN_COMMIT,binding_sha256=binding_sha,results=results,accepted=accepted,
        pair_execution_authorized=False)
        and values['private/screen-seal.local.json']==dict(schema='teacher-f-screen-seal-v1',
        registration_commit=SCREEN_COMMIT,binding_sha256=binding_sha,
        screen_result_sha256=sha(raws['private/screen-result.local.json']),
        projection_sha256=sha(raws['private/projected-summary.local.json']),accepted=accepted,
        pair_execution_authorized=False))
    closed=values['private/closed.local.json']
    require(closed.get('schema')=='teacher-f-screen-closed-v1' and closed.get('registration_commit')==SCREEN_COMMIT
        and closed.get('binding_sha256')==binding_sha and closed.get('outcome')=='completed'
        and closed.get('error_kind') is None and closed.get('accepted_sources')==8
        and closed.get('active_index') is None and closed.get('child_cleanup_receipts')==jobs
        and closed.get('retry_allowed') is False)
    projection=contract.screen_projection(results)
    require(contract.same(projection,values['private/projected-summary.local.json'])
        and contract.same(projection,proof['summary']))
    cases=ordered_cases(results,contract)
    context=modules['f_sources'].open_context(guard=guard,
        expected_identities=binding['completed_sources'],verify_game_files=verify_game_files)
    try:
        plans=context.private_source_plans()
        require([p.document() for p in plans]==binding['source_plans'])
        for index,(result,plan) in enumerate(zip(results,plans)):
            require(set(result)=={'source','source_plan','coverage','candidate'}
                and contract.source_index(result['source'])==index
                and result['source_plan']==plan.document()
                and result['coverage']['source_identity_sha256']==plan.identity_sha256)
            case=modules['f_worker'].validate_case(result['candidate'],context)
            require(case['source_identity_sha256']==plan.identity_sha256
                and result['coverage']['selection_plan'][case['window']][case['rank']]
                    ==dict(step=case['step'],ordinal=case['ordinal']))
        identity=dict(public=proof['identity'],files=witness['files'],witness_sha256=sha(witness_raw),
            runtime_commitment_sha256=witness['commitment_sha256'],completed_sources=context.identities,
            source_plans=[p.document() for p in plans],cases=cases,
            default_sha256=binding['default_sha256'])
        require(sha(infra.read(GAME/'main.py'))==identity['default_sha256'])
        return Evidence(context,identity,cases,list(plans),projection)
    except BaseException:
        try:context.close()
        except BaseException:pass
        raise
