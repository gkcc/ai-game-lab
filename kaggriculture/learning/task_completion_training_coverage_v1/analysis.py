"""Fixed metadata-only sample of a fully decoded, previously consumed cache."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from learning.task_choice_v1.features import OPS
from learning.task_completion_bc_v1 import numeric
from .experiment import OUT, CACHE, ready, verify_commit, deadline_check, read, sha, write_new


class ReceiptFailure(RuntimeError):
    def __init__(self, primary_error_kind, receipt_error_kind):
        super().__init__('Coverage failure receipt could not be saved')
        self.primary_error_kind,self.receipt_error_kind = primary_error_kind,receipt_error_kind


def validate_segments(segments):
    if not isinstance(segments,list) or not segments: raise ValueError('Invalid source segments')
    offset, seen = 0,set()
    for segment in segments:
        if (not isinstance(segment,dict) or set(segment) != {'game','arm','seed','count','offset'}
            or not isinstance(segment['game'],str) or not segment['game'] or segment['game'] in seen
            or not isinstance(segment['arm'],str) or not segment['arm']
            or type(segment['seed']) is not int or not 0 <= segment['seed'] <= 0xffffffff
            or type(segment['count']) is not int or segment['count'] <= 0
            or type(segment['offset']) is not int or segment['offset'] != offset):
            raise ValueError('Invalid source segment')
        seen.add(segment['game'])
        offset += segment['count']
        if offset > 16000: raise ValueError('Group budget exceeded')
    return offset


def deterministic_select(source_segments, limit=200):
    total = validate_segments(source_segments)
    if type(limit) is not int or limit <= 0 or limit > 200 or total < limit: raise ValueError('Invalid selection size')
    identities = []
    for segment in source_segments:
        for ordinal in range(segment['count']):
            game = segment['game']
            payload = json.dumps([game,ordinal],ensure_ascii=False,separators=(',',':')).encode('utf-8')
            identities.append((hashlib.sha256(payload).digest(),game,ordinal))
    identities.sort()
    return [dict(game=game,local_ordinal=ordinal) for _,game,ordinal in identities[:limit]]


def coverage(arrays, schema, segments, positions):
    names = schema['feature_names']
    required = ['unit.index','unit.x','unit.y','target.x','target.y','task.distance'] + ['task.op.'+op for op in OPS]
    if not isinstance(names,list) or len(names) != 98 or any(names.count(n) != 1 for n in required):
        raise ValueError('Invalid feature schema')
    indexes = {name:names.index(name) for name in required}
    by_game = {s['game']:s for s in segments}
    aggregate = dict(source_arms={},selected_operations={},statuses={},distance_gap={},shortest_ties={})
    counters = {key:Counter() for key in aggregate}
    joint = Counter()
    for position in positions:
        segment = by_game[position['game']]
        i = segment['offset'] + position['local_ordinal']
        start,end = int(arrays['offsets'][i]),int(arrays['offsets'][i+1])
        rows, ops = arrays['candidates'][start:end],arrays['ops'][start:end]
        chosen = int(arrays['chosen'][i])
        if len(rows) == 0 or not 0 <= chosen < len(rows) or len(ops) != len(rows): raise ValueError('Invalid chosen or candidates')
        required_values = rows[:,[indexes[n] for n in required]]
        if not np.isfinite(required_values).all() or not np.equal(required_values,np.rint(required_values)).all():
            raise ValueError('Invalid integer feature')
        onehot = rows[:,[indexes['task.op.'+op] for op in OPS]]
        if np.any(ops >= len(OPS)) or not np.array_equal(onehot,np.eye(len(OPS),dtype=np.float32)[ops]):
            raise ValueError('Operation identity differs')
        ux,uy = rows[:,indexes['unit.x']],rows[:,indexes['unit.y']]
        tx,ty = rows[:,indexes['target.x']],rows[:,indexes['target.y']]
        distances = rows[:,indexes['task.distance']]
        if np.any(distances < 0) or not np.array_equal(distances,np.abs(ux-tx)+np.abs(uy-ty)):
            raise ValueError('Distance semantics differ')
        units = rows[:,indexes['unit.index']]
        if np.any(units < 0): raise ValueError('Invalid unit index')
        mask = (tx == tx[chosen]) & (ty == ty[chosen]) & (ops == ops[chosen])
        peers = np.flatnonzero(mask)
        counters['source_arms'][segment['arm']] += 1
        counters['selected_operations'][OPS[int(ops[chosen])]] += 1
        status = int(arrays['status'][i])
        if status not in (0,1): raise ValueError('Invalid status')
        status_name = ('completed_observed','unknown')[status]
        counters['statuses'][status_name] += 1
        joint[(segment['arm'],OPS[int(ops[chosen])],status_name)] += 1
        if len(np.unique(units[peers])) >= 2:
            minimum = int(np.min(distances[peers]))
            counters['distance_gap'][str(int(distances[chosen])-minimum)] += 1
            counters['shortest_ties'][str(len(np.unique(units[peers][distances[peers] == minimum])))] += 1
    result = {key:dict(sorted(value.items())) for key,value in counters.items()}
    result['arm_operation_status'] = [dict(arm=arm,operation=operation,status=status,count=count)
        for (arm,operation,status),count in sorted(joint.items())]
    return result


def run(out=OUT):
    out = Path(out).resolve()
    cfg = ready(out)
    verify_commit(out)
    limits = cfg['limits']
    expected = dict(max_archive_bytes=256*1024**2,max_uncompressed_bytes=512*1024**2,max_groups=16000,max_candidates=1200000,max_selected=200,seconds=300)
    if limits != expected: raise ValueError('Limits changed')
    positions_path = out/'sample-positions.local.json'
    if sha(positions_path) != cfg['sample_positions_sha256']: raise ValueError('Sample identity changed')
    positions = read(positions_path)
    if positions != deterministic_select(cfg['source_segments'],200): raise ValueError('Fixed sample changed')
    if sha(out/'schema.json') != cfg['schema_sha256']: raise ValueError('Schema identity changed')
    if (out/'analysis-started.json').exists(): raise ValueError('Attempt already consumed')
    marker = dict(created=False,complete=False)
    hash_started = decode_started = decode_completed = False
    started = time.monotonic()
    def check():
        deadline_check()
        if time.monotonic()-started > limits['seconds']: raise TimeoutError('Coverage budget exceeded')
    try:
        path = (out/'analysis-started.json').resolve()
        if not path.is_relative_to(OUT): raise ValueError('Marker outside experiment')
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            marker['created'] = True
            json.dump(dict(attempt=1,retry_allowed=False,freeze_sha256=sha(out/'freeze.json')),stream,indent=2)
            stream.write('\n')
        marker['complete'] = True
        check()
        if CACHE.stat().st_size > limits['max_archive_bytes']: raise ValueError('Archive budget exceeded')
        hash_started = True
        if sha(CACHE) != cfg['cache_sha256']: raise ValueError('Cache identity changed')
        check()
        with zipfile.ZipFile(CACHE) as archive:
            if sum(info.file_size for info in archive.infolist()) > limits['max_uncompressed_bytes']:
                raise ValueError('Declared uncompressed budget exceeded')
        check()
        decode_started = True
        arrays = numeric.read(CACHE,limits['max_groups'],limits['max_candidates'])
        decode_completed = True
        if sha(CACHE) != cfg['cache_sha256']: raise ValueError('Cache changed during decode')
        check()
        total = validate_segments(cfg['source_segments'])
        if len(arrays['shared']) != total: raise ValueError('Segment count differs')
        for segment in cfg['source_segments']:
            if np.any(arrays['seed'][segment['offset']:segment['offset']+segment['count']] != segment['seed']):
                raise ValueError('Segment seed differs')
        result = coverage(arrays,read(out/'schema.json'),cfg['source_segments'],positions)
        check()
        output = dict(passed=True,selected_groups=len(positions),aggregates=result,freeze_sha256=sha(out/'freeze.json'),
            sample_positions_sha256=cfg['sample_positions_sha256'],cache_sha256=cfg['cache_sha256'],method_sha256=sha(Path(__file__)),
            full_cache_hash_reads=2,full_npz_decode_passes=1,full_cache_validation_and_copies=True,
            physical_reads='Registration hash, execution before/after hashes, NPZ directory reads and one full decode',
            source_identity='Segment source depends on prior collector order and counts, not independent row IDs',
            interpretation='Observed Manhattan opportunities only; not teacher optimality, actual path length or economic causality',
            joint_count_scope='Observed nonzero arm-operation-status combinations only; omitted combinations have count zero; no rates or zero-denominator ratios computed',
            new_games=0,training_updates=0,model_inference=0,retry_allowed=False)
        write_new(out/'analysis-result.json',output)
        return output
    except BaseException as exc:
        if marker['created']:
            try:
                write_new(out/'analysis-failure.local.json',dict(error_kind=type(exc).__name__,marker=marker,
                    hash_started=hash_started,decode_started=decode_started,decode_completed=decode_completed,
                    partial_decode='unknown' if decode_started and not decode_completed else 0,
                    retry_allowed=False,partial_work_not_refunded=True))
            except BaseException as receipt:
                raise ReceiptFailure(type(exc).__name__,type(receipt).__name__) from None
        raise
