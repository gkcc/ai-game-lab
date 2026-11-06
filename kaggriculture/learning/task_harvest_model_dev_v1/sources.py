"""Authenticate existing identities and extract the four already frozen inputs.

No model construction or official execution. Stream parsing uses the published
bounded CRC/EOF reader, never scans for replacement points or future labels.
"""
from copy import deepcopy

from learning.task_teacher_fresh_panel_v1 import sources as completed
from learning.task_harvest_trajectory_v1 import runner as archive
from learning.task_harvest_trajectory_v1 import contract as archive_contract
from learning.task_harvest_unexpressed_v1.selection import validate_target
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from . import contract


def source_context(guard, *, full=True):
    # False is only for a live, authenticated child whose parent already bound
    # every compressed identity. The coordinator enforces that ownership gate.
    return completed.load(guard=guard, verify_game_files=full)


def archive_metadata(infra, guard):
    value = archive.archive_metadata(infra, archive_contract, guard)
    contract.require(value['small_originals_authenticated']==41 and value['gzip_body_reads']==0)
    for case, expected in zip(value['cases'], contract.sources()):
        contract.require(case['source']==expected)
        target=validate_target(case['target'])
        contract.require(target['side']==expected['side'])
    return value


def required_assets(context, driver, game_root, guard):
    """Hash only the used baseline/crop, rivals and existing Worker."""
    assets={}
    for manifest in context.manifests.values():
        panel=manifest.panel
        for path,digest in (panel.models[0],panel.crop):
            guard()
            contract.require(driver._hash(game_root,game_root/path,driver.MAX_MODEL)==digest)
            if path in assets:contract.require(assets[path]==digest)
            assets[path]=digest
        for path,digest in manifest.opponent_sources:
            guard()
            contract.require(driver._hash(game_root,game_root/path,driver.MAX_SOURCE)==digest)
            if path in assets:contract.require(assets[path]==digest)
            assets[path]=digest
        path='learning/task_anchor_once_v1/games.py'
        guard()
        contract.require(driver._hash(game_root,game_root/path,driver.MAX_SOURCE)==panel.worker_sha)
        if path in assets:contract.require(assets[path]==panel.worker_sha)
        assets[path]=panel.worker_sha
    return assets


def select_point(events, target, *, role, task, crop):
    """Select the exact registered ordinal, consuming the stream through EOF."""
    target=validate_target(target)
    found=None
    active=None
    closed=0
    for event in events:
        kind=event.get('kind')
        if kind=='arm_start':
            contract.require(active is None and closed<2 and event.get('arm')==('control','intervene')[closed])
            active=event['arm']
        elif kind=='arm_end':
            contract.require(active is not None and event.get('arm')==active)
            active=None
            closed+=1
        else:
            contract.require(kind=='turn' and active is not None and event.get('arm')==active)
        if (kind!='turn' or event.get('arm')!='control' or event.get('scope')!='suffix'
                or event.get('step')!=target['step']):
            continue
        contract.require(found is None)
        trace=event['trace']
        contract.require(trace['step']==target['step'] and type(trace['records']) is list
            and len(trace['records'])>target['ordinal'])
        inputs=trace['records'][target['ordinal']]['inputs']
        contract.require(ordered_hash(inputs)==target['input_sha256']
            and inputs['observation']['player']==target['side']
            and inputs['observation']['step']==target['step']
            and event['request']==event['original_request'] and event['patch'] is None
            and ordered_hash(trace['final_reservations']['unit_actions'])==target['unit_requests_sha256'])
        request=event['request']
        units=trace['final_reservations']['unit_actions']
        contract.require(0<=target['unit']<len(units)
            and ordered_hash(units[target['unit']])==ordered_hash(target['original_unit_request'])
            and [request['farmer'],*request['hands']]==units
            and trace['task_checkpoint_sha256']==task[1] and trace['checkpoint_sha256']==crop[1])
        found=dict(role=role,target=deepcopy(target),inputs=deepcopy(inputs),
            original_request=deepcopy(request),original_trace=deepcopy(trace),
            task=list(task),crop=list(crop))
    contract.require(found is not None and active is None and closed==2)
    return found


def points(infra, archive_identity, context, guard, emit):
    """One pass per sealed four-event stream; data stay in the new private run."""
    result=[]
    aggregate={'decoded_bytes':0}
    for case,source in zip(archive_identity['cases'],contract.sources()):
        role=source['role'];manifest=context.manifests[source['cohort']]
        expected=archive_identity['files']['stage/'+role+'/events.jsonl.gz']
        path=archive.PRIOR_STAGE/role/'events.jsonl.gz'
        contract.require(archive.regular(path)==expected['bytes'])
        with path.open('rb') as raw:
            stream=archive.EventStream(raw,expected,archive_contract.LIMITS,aggregate,infra.decode,guard)
            point=select_point(stream,case['target'],role=role,
                task=manifest.panel.models[0],crop=manifest.panel.crop)
            receipt=stream.receipt()
        contract.require(receipt['lines']==2*(case['target']['step']+50))
        emit(role+'.point.local.json',point)
        emit(role+'.input-stream.local.json',dict(role=role,**receipt))
        result.append(point)
    contract.require(len(result)==4)
    return result
