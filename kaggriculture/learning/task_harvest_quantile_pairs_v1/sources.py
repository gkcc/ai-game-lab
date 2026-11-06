"""Two sealed ancestors, four source scans, twelve frozen point slots."""
from copy import deepcopy
import json
from pathlib import Path

from learning.task_harvest_return_pairs_v1 import sources as previous
from learning.task_harvest_return_pairs_v1.selection import validate_target
from learning.task_harvest_window_v1.worker import _stable,trace_factory
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from . import contract
from .selection import select_quantiles

REPO,GAME=previous.REPO,previous.GAME
PAIR_PRIVATE=REPO.parent/'private/task-harvest-return-pairs-v1-20261106'
PAIR_STAGE=GAME/'artifacts/task-harvest-return-pairs-v1-20261106'
PAIR_WITNESS=PAIR_PRIVATE.parent/(PAIR_PRIVATE.name+'-publication-witness.local.json')
PAIR_REG=REPO/'review/evidence/task-harvest-return-pairs-v1-20261106/registration.json'
PAIR_RESULTS=REPO/'review/evidence/task-harvest-return-pairs-v1-result-20261106'
ROOTS=dict(private=PAIR_PRIVATE,stage=PAIR_STAGE)
canonical,digest=previous.canonical,previous.digest


def inventory(witness):
    contract.require(type(witness) is dict and set(witness)=={'schema','registration_commit','domain',
        'files','bytes_total','commitment_sha256'}
        and witness['schema']=='task-harvest-return-pairs-publication-witness-v1'
        and witness['registration_commit']==contract.PAIR_REG_COMMIT
        and witness['domain']=='task-harvest-return-pairs-v1'
        and witness['commitment_sha256']==contract.PAIR_COMMITMENT
        and type(witness['files']) is list and len(witness['files'])==contract.PAIR_FILES
        and type(witness['bytes_total']) is int and witness['bytes_total']==contract.PAIR_BYTES)
    files={}
    for row in witness['files']:
        contract.require(type(row) is dict and set(row)=={'root','path','bytes','sha256'}
            and row['root'] in ROOTS and type(row['bytes']) is int and 0<=row['bytes']<=192*1024**2
            and type(row['sha256']) is str and len(row['sha256'])==64
            and all(c in '0123456789abcdef' for c in row['sha256']))
        key=(row['root'],previous._relative(row['path']))
        contract.require(key not in files);files[key]=row
    contract.require(list(files)==sorted(files) and sum(r['bytes'] for r in files.values())==contract.PAIR_BYTES
        and {k:sum(label==k for label,_ in files) for k in ROOTS}==dict(private=20,stage=45))
    return files


def archive(infra,dm,guard,*,full):
    """Byte authentication only for gzip bodies; no Replay or model decoding."""
    base=previous.archive(infra,dm,guard,full=full)
    raw=dm._read(PAIR_PRIVATE.parent,PAIR_WITNESS,1024**2)
    witness=infra.decode(raw);files=inventory(witness)
    def read(label,name,limit=8*1024**2):
        guard();entry=files[(label,name)]
        value=dm._read(ROOTS[label],ROOTS[label]/name,limit)
        contract.require(len(value)==entry['bytes'] and digest(value)==entry['sha256'])
        return value
    binding_raw=read('private','binding.local.json');binding=infra.decode(binding_raw)
    reg=infra.decode(infra.read(PAIR_REG));cert=infra.decode(infra.read(PAIR_RESULTS/'certificate.json'))
    contract.require(digest(binding_raw)==reg['binding_commitment_sha256']
        and digest(bytes.fromhex(binding['salt'])+b'\0'+witness['domain'].encode()+b'\0'+canonical(witness['files']))
            ==witness['commitment_sha256']==cert['evidence']['salted_commitment_sha256']
        and cert['registration_commit']==contract.PAIR_REG_COMMIT and cert['execution_validation_passed'] is True
        and canonical(base)==canonical(binding['archive']))
    if full:
        for label,boundary in ROOTS.items():
            actual={p.relative_to(boundary).as_posix() for p in boundary.rglob('*') if p.is_file()}
            contract.require(actual=={name for r,name in files if r==label})
            for name in sorted(actual):
                guard();entry=files[(label,name)]
                contract.require((boundary/name).stat().st_size==entry['bytes']
                    and dm._hash(boundary,boundary/name,192*1024**2)==entry['sha256'])
    closed=infra.decode(read('private','closed.local.json'));result=infra.decode(read('private','result.local.json'))
    projected=read('private','projected-summary.local.json');published=infra.read(PAIR_RESULTS/'summary.json')
    contract.require(closed['outcome']=='completed' and closed['error_kind'] is None and closed['accepted_cases']==4
        and result['registration_commit']==closed['registration_commit']==contract.PAIR_REG_COMMIT
        and result['binding_sha256']==closed['binding_sha256']==digest(binding_raw)
        and len(result['accepted'])==4 and len(closed['child_cleanup_receipts'])==4
        and result['projection_sha256']==digest(projected) and cert['public_summary_sha256']==digest(published)
        and canonical(infra.decode(projected))==canonical(infra.decode(published)))
    references=[]
    for source,accepted in zip(base['cases'],result['accepted']):
        origin=source['role'];doc_raw=read('stage',origin+'/case.local.json');doc=infra.decode(doc_raw)
        selected_raw=read('private',origin+'.selection.local.json');selected=infra.decode(selected_raw)
        sample=files[('stage',origin+'/sample.local.json')]
        contract.require(digest(doc_raw)==accepted['document_sha256'] and doc['passed'] is True
            and canonical(doc['source'])==canonical(source) and canonical(doc['row'])==canonical(accepted['row'])
            and digest(selected_raw)==binding['selections'][origin]
            and ordered_hash(doc['target'])==ordered_hash(selected['selection']['target'])
            and doc['row']['status']=='paired' and doc['row']['technical_passed'] is True
            and doc['sample']['path']=='sample.local.json' and doc['sample']['sha256']==sample['sha256']
            and doc['sample']['bytes']==sample['bytes'])
        references.append(dict(origin=origin,target=validate_target(doc['target']),row=deepcopy(doc['row']),
            sample=[(PAIR_STAGE/(origin+'/sample.local.json')).relative_to(GAME).as_posix(),sample['sha256'],sample['bytes']]))
    contract.require([c['role'] for c in base['cases']]==list(contract.ORIGINS)
        and len({c['row']['seed'] for c in base['cases']})==4
        and len({tuple(c['task']) for c in base['cases']})==1)
    guard()
    return dict(base,pair_witness_sha256=digest(raw),pair_commitment_sha256=witness['commitment_sha256'],
        pair_binding_sha256=digest(binding_raw),references=references)


def assets_verify(dm,archive_identity,guard):return previous.assets_verify(dm,archive_identity,guard)


def select_sources(dm,archive_identity,guard,emit):
    output=[]
    for source,reference in zip(archive_identity['cases'],archive_identity['references']):
        history=[];scans=[]
        def indexed(rows):
            for row in rows:
                history.append(dict(step=row['step'],stable_trace_sha256=ordered_hash(_stable(row)),
                    model_counts=deepcopy(row['model_counts']),turn_model_counts=deepcopy(row['turn_model_counts'])))
                yield row
        with trace_factory(dm,GAME,GAME/source['trace'][0],source['trace'][1],guard,scans)() as rows:
            selection=select_quantiles(indexed(rows),side=source['side'],task_sha=source['task'][1],
                crop_sha=source['crop'][1],old_target=reference['target'],guard=guard)
        contract.require(len(history)==719 and scans==[dict(lines=719,complete=True,closed=True)])
        value=dict(origin=source['role'],source=deepcopy(source),quantiles=selection,trace_index=history,trace_scans=scans)
        emit(source['role']+'.selection.local.json',value);output.append(value)
    return output


def selected_point(document,point):
    contract.require(point in contract.POINTS)
    items=[v for v in document['quantiles']['points'] if v['label']==point]
    contract.require(len(items)==1)
    item=items[0]
    return dict(role=document['origin']+'-'+point,origin=document['origin'],point=point,
        split=contract.split(document['origin']),source=deepcopy(document['source']),
        selection=dict(target=deepcopy(item['target']),reason=item['reason'],index=item['index'],
            counts=deepcopy(document['quantiles']['counts'])),
        trace_index=deepcopy(document['trace_index']),trace_scans=deepcopy(document['trace_scans']))


def plans(documents):
    contract.require([v['origin'] for v in documents]==list(contract.ORIGINS))
    result=[]
    for document in documents:
        for point in contract.POINTS:
            selected=selected_point(document,point);target=selected['selection']['target']
            contract.require(target is None or target['step']>=240)
            result.append(dict(role=selected['role'],origin=selected['origin'],point=point,split=selected['split'],
                index=selected['selection']['index'],reason=selected['selection']['reason'],
                target_sha256=None if target is None else ordered_hash(target),
                official_actions=0 if target is None else 2*(719-target['step'])))
    contract.require([v['role'] for v in result]==list(contract.ROLES)
        and sum(v['official_actions'] for v in result)<=contract.LIMITS['max_official_action_transitions'])
    return result


def reference_samples(dm,archive_identity,guard):
    result=[]
    for source,reference in zip(archive_identity['cases'],archive_identity['references']):
        guard();relative,expected,size=reference['sample']
        raw=dm._read(GAME,GAME/relative,8*1024**2)
        contract.require(len(raw)==size and digest(raw)==expected)
        sample=json.loads(raw)
        result.append(dict(origin=source['role'],point='Q0',split=contract.split(source['role']),
            seed=source['row']['seed'],target=deepcopy(reference['target']),sample=sample))
    return result
