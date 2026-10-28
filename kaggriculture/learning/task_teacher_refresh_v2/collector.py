"""Bounded teacher selections from eight attested development traces.

Retained examples deliberately oversample DIG and FERTILIZE. They are neither
population frequencies nor evidence of official execution or task completion.
"""
from pathlib import Path
from learning.task_action_impact_diagnostic_v1.analysis import Meter
from learning.task_action_observation_v1.analysis import raw_path
from learning.task_completion_bc_v1.data import collect_game
from learning.task_teacher_choice_bc_v1.collect import bounded_turns, pack, selected
from learning.task_teacher_choice_bc_v1.extract import extract
from learning.task_choice_v1.features import OPS

RARE=frozenset(('DIG','FERTILIZE'))
LIMIT_KEYS=frozenset(('compressed_bytes','decoded_bytes','max_line_bytes','max_groups','max_candidates',
                      'max_candidates_per_group','max_total_callbacks','max_total_candidate_rows'))


def digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def validate_sources(sources,attested,seeds):
    if (not isinstance(seeds,list) or len(seeds)!=2 or any(type(s) is not int or not 0<=s<2**32 for s in seeds)
        or len(set(seeds))!=2 or not isinstance(sources,list) or len(sources)!=8 or sources!=attested):
        raise ValueError('Invalid source attestations')
    expected=[(seed,opponent,side) for seed in seeds for opponent in ('round2','market') for side in (0,1)]
    for source,(seed,opponent,side) in zip(sources,expected):
        game=f'dev-{seed}-{opponent}-{side}-teacher_task'
        if (not isinstance(source,dict) or set(source)!={'game','seed','opponent','side','trace','replay_sha256','audit_sha256'}
            or type(source['seed']) is not int or type(source['side']) is not int
            or (source['game'],source['seed'],source['opponent'],source['side'])!=(game,seed,opponent,side)
            or not digest(source['replay_sha256']) or not digest(source['audit_sha256'])):
            raise ValueError('Invalid source identity')
        trace=source['trace']
        if (not isinstance(trace,dict) or set(trace)!={'path','size','sha256'}
            or trace['path']!='traces/'+game+'.jsonl.gz' or type(trace['size']) is not int or trace['size']<0
            or not digest(trace['sha256'])): raise ValueError('Invalid trace identity')


def collect(source_root,sources,attested_sources,seeds,schema,limits,check):
    """Return arrays only after all sources pass; caller authenticates attestations."""
    validate_sources(sources,attested_sources,seeds)
    if (not isinstance(limits,dict) or set(limits)!=LIMIT_KEYS
        or any(type(limits[k]) is not int or limits[k]<=0 for k in LIMIT_KEYS)
        or limits['max_candidates_per_group']>65536 or not callable(check)):
        raise ValueError('Invalid resource limits')
    meter=Meter(limits['compressed_bytes'],limits['decoded_bytes'])
    groups=[]; rows=0; total_callbacks=0; total_rows=0; stats=[]
    for seed in seeds:
        for opponent in ('round2','market'):
            stats.append(dict(seed=seed,opponent=opponent,callbacks=0,retained=0,
                chosen={op:0 for op in OPS},retained_chosen={op:0 for op in OPS},
                candidate_rows={op:0 for op in OPS},candidate_callbacks={op:0 for op in OPS}))
    by_key={(s['seed'],s['opponent']):s for s in stats}
    for source in sources:
        check(); game=dict(key=source['game'],side=source['side'],seed=source['seed'])
        path=raw_path(Path(source_root),source['trace'],'trace')
        stream=bounded_turns(path,source['trace'],meter,check,limits['max_line_bytes'])
        def checked_turns():
            nonlocal rows,total_callbacks,total_rows
            for turn in stream:
                check()
                for ordinal,record in enumerate(turn['records']):
                    if not isinstance(record,dict) or not isinstance(record.get('inputs'),dict):
                        raise ValueError('Invalid callback structure')
                    candidates=record['inputs'].get('candidates')
                    if not isinstance(candidates,list) or not 1<=len(candidates)<=limits['max_candidates_per_group']:
                        raise ValueError('Invalid full-stream candidate count')
                    total_callbacks+=1; total_rows+=len(candidates)
                    if total_callbacks>limits['max_total_callbacks'] or total_rows>limits['max_total_candidate_rows']:
                        raise ValueError('Full-stream budget exceeded')
                    if (not isinstance(record,dict) or record.get('source')!='teacher'
                        or not isinstance(record.get('teacher_label'),str) or record['teacher_label']!=record.get('chosen')):
                        raise ValueError('Teacher choice differs from actor')
                    item=extract(record,schema,side=source['side'],step=turn['step'],
                                 max_candidates=limits['max_candidates_per_group'])
                    ops=item['candidate_ops']; choice=ops[item['teacher_choice_label']]
                    entry=by_key[(source['seed'],source['opponent'])]
                    entry['callbacks']+=1; entry['chosen'][choice]+=1
                    for op in OPS:
                        count=ops.count(op); entry['candidate_rows'][op]+=count
                        entry['candidate_callbacks'][op]+=int(count>0)
                    if choice in RARE or selected(source['game'],turn['step'],ordinal):
                        rows+=len(item['candidates'])
                        if len(groups)+1>limits['max_groups'] or rows>limits['max_candidates']:
                            raise ValueError('Retained sample budget exceeded')
                        groups.append(dict(item,seed=source['seed']))
                        entry['retained']+=1; entry['retained_chosen'][choice]+=1
                yield turn
        try:
            _,unused=collect_game(game,[],checked_turns(),schema)
            if unused: raise ValueError('Unexpected episode groups')
        finally:
            stream.close()
    snapshot=meter.snapshot()
    if (snapshot['files_opened']!=8 or snapshot['files_completed']!=8
        or snapshot['raw_decode_passes_per_file']!=1): raise ValueError('Incomplete source integrity')
    arrays=pack(groups,limits['max_groups'],limits['max_candidates'])
    return arrays,dict(by_seed_opponent=stats,total_callbacks=total_callbacks,total_candidate_rows=total_rows,
        retained_groups=len(groups),retained_candidate_rows=rows,meter=snapshot,
        sampling='DIG/FERTILIZE selected labels always retained; other labels use registered identity hash at 12.5%',
        interpretation='Retained distribution is intentionally altered; selections do not establish execution or completion')
