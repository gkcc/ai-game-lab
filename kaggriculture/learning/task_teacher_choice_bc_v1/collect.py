"""Single-pass bounded teacher-label collector; no output before all games pass."""
import hashlib
import json
import numpy as np
from learning.task_action_impact_diagnostic_v1.analysis import Meter, decoded_chunks
from learning.task_action_observation_v1.analysis import raw_path
from learning.task_completion_bc_v1.data import collect_game
from learning.task_choice_v1.features import OPS
from .extract import extract

SAMPLING_ALGORITHM = 'sha256-compact-json-game-step-ordinal-first16-lt8192'


def selected(game,step,ordinal):
    if not isinstance(game,str) or not game or type(step) is not int or not 0 <= step < 719 or type(ordinal) is not int or ordinal < 0:
        raise ValueError('Invalid sampling identity')
    payload = json.dumps([game,step,ordinal],ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:2],'big') < 8192


def pack(groups,max_groups,max_candidates):
    if type(max_groups) is not int or type(max_candidates) is not int or min(max_groups,max_candidates) < 0:
        raise ValueError('Invalid limits')
    if not isinstance(groups,list) or len(groups) > max_groups: raise ValueError('Invalid groups')
    shared,local,labels,ops,seeds,offsets = [],[],[],[],[],[0]
    for g in groups:
        if set(g) != {'shared','candidates','teacher_choice_label','candidate_ops','seed'}: raise ValueError('Invalid group fields')
        s,c = np.asarray(g['shared']),np.asarray(g['candidates'])
        label,operations = g['teacher_choice_label'],g['candidate_ops']
        if (s.dtype != np.float32 or s.shape != (275,) or c.dtype != np.float32 or c.ndim != 2
            or c.shape[1] != 98 or not len(c) or not np.isfinite(s).all() or not np.isfinite(c).all()
            or type(label) is not int or not 0 <= label < len(c) or len(operations) != len(c)
            or type(g['seed']) is not int or not 0 <= g['seed'] <= 0xffffffff
            or any(op not in OPS for op in operations) or offsets[-1]+len(c) > max_candidates):
            raise ValueError('Invalid numeric group')
        shared.append(s); local.append(c); labels.append(label); seeds.append(g['seed']); ops.extend(OPS.index(op) for op in operations)
        offsets.append(offsets[-1]+len(c))
    return dict(shared=np.asarray(shared,dtype=np.float32).reshape(-1,275),
        candidates=np.concatenate(local) if local else np.empty((0,98),np.float32),
        offsets=np.asarray(offsets,np.int64),teacher_choice_label=np.asarray(labels,np.int64),
        ops=np.asarray(ops,np.uint8),seed=np.asarray(seeds,np.uint32))


def bounded_turns(path,identity,meter,check,max_line_bytes):
    with decoded_chunks(path,identity,meter) as chunks:
        pending = bytearray()
        for block in chunks:
            check()
            pending.extend(block)
            while True:
                end = pending.find(b'\n')
                if end < 0: break
                if end == 0 or end > max_line_bytes: raise ValueError('Invalid trace line')
                yield json.loads(bytes(pending[:end]))
                del pending[:end+1]
            if len(pending) > max_line_bytes: raise ValueError('Trace line exceeds budget')
        if pending: raise ValueError('Missing trace newline')


def collect(source_root, sources, attested_sources, expected_schema, limits, check):
    """Consume all eight traces once; caller independently binds attestations via ready.

    sources and attested_sources are identical ordered metadata: game, side, seed,
    trace {path,size,sha256}, replay_sha256, audit_sha256. No replay is opened.
    """
    if (not isinstance(sources,list) or len(sources) != 8 or sources != attested_sources
        or len({s['game'] for s in sources}) != 8): raise ValueError('Source attestations differ')
    keys = {'compressed_bytes','decoded_bytes','max_line_bytes','max_groups','max_candidates','max_candidates_per_group'}
    if set(limits) != keys or any(type(limits[k]) is not int or limits[k] <= 0 for k in keys):
        raise ValueError('Invalid resource limits')
    def digest(x): return isinstance(x,str) and len(x)==64 and all(ch in '0123456789abcdef' for ch in x)
    for source in sources:
        if (set(source) != {'game','side','seed','trace','replay_sha256','audit_sha256'}
            or not digest(source['replay_sha256']) or not digest(source['audit_sha256'])
            or not isinstance(source['trace'],dict) or not digest(source['trace'].get('sha256'))):
            raise ValueError('Invalid source identity')
    meter = Meter(limits['compressed_bytes'],limits['decoded_bytes'])
    groups,rows = [],0
    for source in sources:
        check()
        game = dict(key=source['game'],side=source['side'],seed=source['seed'])
        path = raw_path(source_root,source['trace'],'trace')
        stream = bounded_turns(path,source['trace'],meter,check,limits['max_line_bytes'])
        def selected_turns():
            nonlocal rows
            for turn in stream:
                check()
                for ordinal,record in enumerate(turn['records']):
                    if selected(source['game'],turn['step'],ordinal):
                        item = extract(record,expected_schema,side=source['side'],step=turn['step'],
                                       max_candidates=limits['max_candidates_per_group'])
                        rows += len(item['candidates'])
                        if len(groups)+1 > limits['max_groups'] or rows > limits['max_candidates']:
                            raise ValueError('Sample resource budget exceeded')
                        groups.append(dict(item,seed=source['seed']))
                yield turn
        try:
            _, unused = collect_game(game,[],selected_turns(),expected_schema)
            if unused: raise ValueError('Unexpected episode groups')
        finally:
            stream.close()
    snapshot = meter.snapshot()
    if snapshot['files_opened'] != 8 or snapshot['files_completed'] != 8 or snapshot['raw_decode_passes_per_file'] != 1:
        raise ValueError('Incomplete source integrity')
    return pack(groups,limits['max_groups'],limits['max_candidates']),snapshot
