"""Pure, bounded development-trajectory sample and paired numeric scoring.

Caller supplies frozen source attestations and a single-pass stream; this
module does not open files or certify source provenance.
"""
import hashlib
import heapq
import json
import numpy as np
from learning.task_choice_v1.features import OPS
from learning.task_completion_bc_v1.data import collect_game
from learning.task_teacher_choice_bc_v1.extract import extract
from learning.task_teacher_choice_bc_v1.collect import bounded_turns
from learning.task_action_impact_diagnostic_v1.analysis import Meter
from learning.task_action_observation_v1.analysis import raw_path

GAMES=('dev-3911611436-market-0-teacher_task','dev-2328454240-market-0-teacher_task')
DOMAIN='task-teacher-dev-deviation-v1:'
PER_GAME=100
MAX_TOTAL_ROWS=120000
MAX_RETAINED_BYTES=256*1024**2
MAX_PER_GAME_BYTES=128*1024**2


def identity_rank(game,step,ordinal):
    if game not in GAMES or type(step) is not int or not 0<=step<719 or type(ordinal) is not int or ordinal<0:
        raise ValueError('Invalid callback identity')
    payload=(DOMAIN+json.dumps([game,step,ordinal],ensure_ascii=True,separators=(',',':'))).encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest(),'big')


def collect_game_sample(game,turns,expected_schema,*,max_candidates_per_group,max_retained_bytes=MAX_PER_GAME_BYTES,check=lambda:None):
    """One pass, retaining only bounded numeric top-100 after complete validation."""
    expected_seed=dict(zip(GAMES,(3911611436,2328454240)))
    if (game.get('key') not in GAMES or game.get('seed') != expected_seed.get(game.get('key')) or type(game.get('seed')) is not int
        or type(game.get('side')) is not int or game['side'] != 0 or type(max_candidates_per_group) is not int or max_candidates_per_group<=0
        or type(max_retained_bytes) is not int or not 0<max_retained_bytes<=MAX_PER_GAME_BYTES):
        raise ValueError('Invalid sampling scope or limits')
    heap=[]; retained=0
    def wrapped():
        nonlocal retained
        for turn in turns:
            check()
            for ordinal,record in enumerate(turn['records']):
                if record['teacher_label'] != record['chosen']: raise ValueError('Teacher trajectory selection changed')
                rank=identity_rank(game['key'],turn['step'],ordinal)
                # Heap keeps largest identity rank at its root; ordinal tie-break
                # makes ordering deterministic even in the event of hash collision.
                identity=(rank,turn['step'],ordinal)
                worst=(-heap[0][0],-heap[0][1],-heap[0][2]) if heap else None
                if len(heap)<PER_GAME or identity < worst:
                    item=extract(record,expected_schema,side=game['side'],step=turn['step'],max_candidates=max_candidates_per_group)
                    size=item['shared'].nbytes+item['candidates'].nbytes
                    previous=heap[0][4] if len(heap)==PER_GAME else 0
                    if retained-previous+size>max_retained_bytes: raise ValueError('Retained sample memory budget exceeded')
                    entry=(-rank,-turn['step'],-ordinal,item,size)
                    if len(heap)==PER_GAME: heapq.heapreplace(heap,entry)
                    else: heapq.heappush(heap,entry)
                    retained += size-previous
            yield turn
    _,unused=collect_game(game,[],wrapped(),expected_schema)
    if unused or len(heap)!=PER_GAME: raise ValueError('Incomplete or insufficient validated sample')
    return [entry[3] for entry in sorted(heap,key=lambda e:(-e[0],-e[1],-e[2]))]


def collect_two(source_root,sources,attested_sources,expected_schema,*,max_candidates_per_group,check):
    """Read two attested trace streams once; caller must verify audit identities."""
    if (not isinstance(sources,list) or sources!=attested_sources or len(sources)!=2
        or [s.get('game') for s in sources]!=list(GAMES)):
        raise ValueError('Source identities differ')
    meter=Meter(32*1024**2,1024**3); samples=[]
    for source in sources:
        def digest(value): return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
        if (set(source)!={'game','side','seed','trace','replay_sha256','audit_sha256'}
            or not isinstance(source['trace'],dict) or not digest(source['replay_sha256']) or not digest(source['audit_sha256'])):
            raise ValueError('Invalid source metadata')
        check(); path=raw_path(source_root,source['trace'],'trace')
        stream=bounded_turns(path,source['trace'],meter,check,16*1024**2)
        try:
            samples.extend(collect_game_sample(dict(key=source['game'],side=source['side'],seed=source['seed']),stream,
                expected_schema,max_candidates_per_group=max_candidates_per_group,check=check))
        finally: stream.close()
    snapshot=meter.snapshot()
    if snapshot['files_opened']!=2 or snapshot['files_completed']!=2 or snapshot['raw_decode_passes_per_file']!=1:
        raise ValueError('Incomplete trace integrity')
    return samples,snapshot


def analyze(samples,new_model,old_model,progress=None):
    if (not isinstance(samples,list) or len(samples)!=200 or not callable(getattr(new_model,'score',None))
        or not callable(getattr(old_model,'score',None)) or (progress is not None and not callable(progress))):
        raise ValueError('Invalid sample or models')
    rows=0; retained=0
    for g in samples:
        if not isinstance(g,dict) or set(g)!={'shared','candidates','teacher_choice_label','candidate_ops'}:
            raise ValueError('Invalid group fields')
        s,c=g['shared'],g['candidates']; label=g['teacher_choice_label']; ops=g['candidate_ops']
        if (not isinstance(s,np.ndarray) or s.dtype!=np.float32 or s.shape!=(275,) or not np.isfinite(s).all()
            or not isinstance(c,np.ndarray) or c.dtype!=np.float32 or c.ndim!=2 or c.shape[1]!=98 or not len(c)
            or not np.isfinite(c).all() or type(label) is not int or not 0<=label<len(c)
            or len(ops)!=len(c) or any(op not in OPS for op in ops)):
            raise ValueError('Invalid numeric group')
        rows+=len(c); retained+=s.nbytes+c.nbytes
    if rows*2>MAX_TOTAL_ROWS or retained>MAX_RETAINED_BYTES: raise ValueError('Forward or memory budget exceeded')
    models={name:{op:dict(count=0,top1=0,rank_sum=0) for op in OPS} for name in ('new','old')}
    paired=dict(both_correct=0,new_only=0,old_only=0,both_wrong=0); completed=0
    for g in samples:
        teacher=g['teacher_choice_label']; matches={}
        for name,model in (('new',new_model),('old',old_model)):
            if progress: progress(dict(forwards_started=completed+1,forwards_completed=completed))
            scores=np.asarray(model.score(g['shared'],g['candidates']),dtype=np.float64)
            completed+=1
            if progress: progress(dict(forwards_started=completed,forwards_completed=completed))
            if scores.shape!=(len(g['candidates']),) or not np.isfinite(scores).all(): raise ValueError('Invalid scores')
            matched=int(np.argmax(scores))==teacher; matches[name]=matched
            rank=1+int(np.count_nonzero(scores>scores[teacher]))+int(np.count_nonzero(scores[:teacher]==scores[teacher]))
            stat=models[name][g['candidate_ops'][teacher]]; stat['count']+=1; stat['top1']+=int(matched); stat['rank_sum']+=rank
        paired['both_correct' if matches['new'] and matches['old'] else 'new_only' if matches['new'] else 'old_only' if matches['old'] else 'both_wrong']+=1
    for by_op in models.values():
        for value in by_op.values():
            value['top1_rate']=value['top1']/value['count'] if value['count'] else None
            value['mean_rank']=value['rank_sum']/value['count'] if value['count'] else None
    return dict(groups=200,cpu_forwards=completed,candidate_rows_per_model=rows,candidate_rows_total=rows*2,models=models,paired=paired,
        interpretation='Development teacher-task trajectories on training seeds; new model trained on other trajectories from these seeds, old exposure may differ. No holdout, generalization or economic causality claim')
