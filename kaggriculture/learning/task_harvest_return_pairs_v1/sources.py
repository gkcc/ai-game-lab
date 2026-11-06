"""Authenticate the four newly completed games, then select without inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath

from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_harvest_window_v1.worker import _stable, trace_factory
from learning.task_pair_dev_games_v1.panel import _model
from . import contract


REPO=Path(__file__).resolve().parents[3]
GAME=REPO.parent.parent/'kaggriculture'
PRIOR_PRIVATE=REPO.parent/'private/task-harvest-model-dev-v1-20261106'
PRIOR_WITNESS=PRIOR_PRIVATE.parent/(PRIOR_PRIVATE.name+'-publication-witness.local.json')
PRIOR_EVIDENCE=REPO/'review/evidence/task-harvest-model-dev-v1-result-20261106'
PRIOR_REG=REPO/'review/evidence/task-harvest-model-dev-v1-20261106/registration.json'
ROOTS={'private':PRIOR_PRIVATE,**{c:GAME/('artifacts/task-harvest-model-dev-'+c.lower()+'-v1-20261106')
                                 for c in ('C01','C02')}}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def digest(raw):return hashlib.sha256(raw).hexdigest()


def _relative(value):
    contract.require(type(value) is str and value and '\\' not in value and ':' not in value
        and all(ord(c)>=32 for c in value))
    path=PurePosixPath(value)
    contract.require(not path.is_absolute() and path.as_posix()==value
        and not any(p in ('.','..') for p in path.parts))
    return value


def inventory(witness):
    """Pure validation before any witness-provided filename is opened."""
    contract.require(type(witness) is dict and set(witness)=={'schema','registration_commit','domain',
        'files','bytes_total','commitment_sha256'}
        and witness['schema']=='task-harvest-model-dev-publication-witness-v1'
        and witness['registration_commit']==contract.PRIOR_REG_COMMIT
        and witness['domain']=='task-harvest-model-dev-v1'
        and witness['commitment_sha256']==contract.PRIOR_COMMITMENT
        and type(witness['files']) is list and len(witness['files'])==contract.PRIOR_FILES
        and type(witness['bytes_total']) is int and witness['bytes_total']==contract.PRIOR_BYTES)
    files={}
    for entry in witness['files']:
        contract.require(type(entry) is dict and set(entry)=={'root','path','bytes','sha256'}
            and entry['root'] in ROOTS and type(entry['bytes']) is int and 0<=entry['bytes']<=128*1024**2
            and type(entry['sha256']) is str and len(entry['sha256'])==64
            and all(c in '0123456789abcdef' for c in entry['sha256']))
        key=(entry['root'],_relative(entry['path']))
        contract.require(key not in files)
        files[key]=entry
    contract.require(list(files)==sorted(files) and sum(e['bytes'] for e in files.values())==contract.PRIOR_BYTES
        and {r:sum(k[0]==r for k in files) for r in ROOTS}=={'private':41,'C01':52,'C02':52})
    return files


def unique_model(assets, expected):
    pairs=[_model({'path':path,'sha256':value}) for path,value in assets.items()
           if value==expected and PurePosixPath(path).suffix=='.npz']
    contract.require(len(pairs)==1,'Frozen model identity is absent or ambiguous')
    return list(pairs[0])


def archive(infra,dm,guard,*,full):
    """Full mode hashes 145 sealed originals; child mode reads bound metadata."""
    contract.require(type(full) is bool and callable(guard))
    raw=dm._read(PRIOR_PRIVATE.parent,PRIOR_WITNESS,1024**2)
    witness=infra.decode(raw);files=inventory(witness)
    def read(label,relative,limit=8*1024**2):
        guard();entry=files[(label,relative)]
        value=dm._read(ROOTS[label],ROOTS[label]/relative,limit)
        contract.require(len(value)==entry['bytes'] and digest(value)==entry['sha256'])
        return value
    reg=infra.decode(infra.read(PRIOR_REG))
    public=infra.decode(infra.read(PRIOR_EVIDENCE/'certificate.json'))
    binding_raw=read('private','binding.local.json');binding=infra.decode(binding_raw)
    salt=binding['salt']
    contract.require(type(salt) is str and len(salt)==64 and all(c in '0123456789abcdef' for c in salt)
        and digest(binding_raw)==reg['binding_commitment_sha256']
        and digest(bytes.fromhex(salt)+b'\0'+witness['domain'].encode()+b'\0'+canonical(witness['files']))
            ==witness['commitment_sha256']==public['evidence']['salted_commitment_sha256']
        and public['registration_commit']==contract.PRIOR_REG_COMMIT
        and public['execution_validation_passed'] is True and public['execution']['complete_games']==4
        and public['execution']['official_main_replays']==4)
    if full:
        for label,boundary in ROOTS.items():
            actual={p.relative_to(boundary).as_posix() for p in boundary.rglob('*') if p.is_file()}
            contract.require(actual=={path for r,path in files if r==label})
            for path in sorted(actual):
                guard();entry=files[(label,path)]
                dm._no_links(boundary,boundary/path)
                contract.require((boundary/path).stat().st_size==entry['bytes']
                    and dm._hash(boundary,boundary/path,128*1024**2)==entry['sha256'])
    closed=infra.decode(read('private','closed.local.json'))
    result=infra.decode(read('private','result.local.json'))
    projected_raw=read('private','projected-summary.local.json')
    public_summary_raw=infra.read(PRIOR_EVIDENCE/'summary.json')
    contract.require(closed['outcome']=='completed' and closed['error_kind'] is None
        and closed['accepted_games']==4 and result['registration_commit']==closed['registration_commit']==contract.PRIOR_REG_COMMIT
        and result['binding_sha256']==closed['binding_sha256']==digest(binding_raw)
        and len(result['accepted'])==4 and len(closed['child_cleanup_receipts'])==4
        and result['projection_sha256']==digest(projected_raw)
        and public['public_summary_sha256']==digest(public_summary_raw)
        and canonical(infra.decode(projected_raw))==canonical(infra.decode(public_summary_raw)))
    assets=binding['assets'];contract.require(type(assets) is dict and len(assets)==5)
    cases=[]
    for source,accepted in zip(contract.sources(),result['accepted']):
        role,cohort,game=source['role'],source['cohort'],source['game']
        actual=infra.decode(read('private',role+'.result.local.json'))
        row=infra.decode(read(cohort,'job-results/'+game+'.row.json'))
        contract.require(canonical(actual)==canonical(accepted) and canonical(row)==canonical(actual['candidate'])
            and all(actual[k]==source[k] for k in ('role','cohort','opponent','side'))
            and all(row[k]==source[k] for k in ('game','seed_alias','opponent','side'))
            and row['policy']=='harvest_model' and row['passed'] is True
            and row['states']==720 and row['action_turns']==719 and row['status']==['DONE','DONE'])
        plans=[p for p in binding['plans'][cohort] if p['game']==game]
        contract.require(len(plans)==1 and all(type(row[k]) is type(v) and row[k]==v for k,v in plans[0].items()))
        paths={'replay':'games/'+game+'/replay.json.gz','trace':'traces/'+game+'.jsonl.gz',
               'rival':'sources/'+source['opponent']+'.py'}
        paired={}
        for kind,key in (('replay','replay_gzip_sha256'),('trace','trace_sha256'),('rival','rival_sha256')):
            path=paths[kind];contract.require(files[(cohort,path)]['sha256']==row[key])
            paired[kind]=[(ROOTS[cohort]/path).relative_to(GAME).as_posix(),row[key]]
        cases.append(dict(source,**paired,task=unique_model(assets,row['task_checkpoint_sha256']),
            crop=unique_model(assets,row['crop_checkpoint_sha256']),row=deepcopy(row)))
    guard()
    return dict(witness_sha256=digest(raw),commitment_sha256=witness['commitment_sha256'],
        prior_binding_sha256=digest(binding_raw),default_sha256=binding['default_sha256'],
        assets=assets,cases=cases)


def assets_verify(dm,archive_identity,guard):
    for relative,expected in archive_identity['assets'].items():
        guard();_relative(relative)
        limit=dm.MAX_MODEL if PurePosixPath(relative).suffix=='.npz' else dm.MAX_SOURCE
        contract.require(dm._hash(GAME,GAME/relative,limit)==expected)
    contract.require(dm._hash(GAME,GAME/'main.py',dm.MAX_SOURCE)==archive_identity['default_sha256'])
    return True


def select_cases(dm,archive_identity,guard,emit):
    """One complete trace pass per case; retain only point and compact index."""
    from .selection import select_target
    selections=[]
    for case in archive_identity['cases']:
        history=[];scans=[]
        def indexed(rows):
            for trace in rows:
                history.append(dict(step=trace['step'],stable_trace_sha256=ordered_hash(_stable(trace)),
                    model_counts=deepcopy(trace['model_counts']),turn_model_counts=deepcopy(trace['turn_model_counts'])))
                yield trace
        with trace_factory(dm,GAME,GAME/case['trace'][0],case['trace'][1],guard,scans)() as rows:
            selected=select_target(indexed(rows),side=case['side'],task_sha=case['task'][1],crop_sha=case['crop'][1],guard=guard)
        contract.require(len(history)==719 and scans==[dict(lines=719,complete=True,closed=True)])
        value=dict(role=case['role'],source=deepcopy(case),selection=selected,trace_index=history,trace_scans=scans)
        emit(case['role']+'.selection.local.json',value)
        selections.append(value)
    return selections
