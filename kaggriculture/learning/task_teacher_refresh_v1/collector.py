"""One-pass extraction from the four accepted teacher-task traces.

Teacher labels describe candidate selection under a mixed controller. They are
not evidence of a completed task, legal execution, or an optimal decision.
"""
from pathlib import Path
from learning.task_action_impact_diagnostic_v1.analysis import Meter
from learning.task_action_observation_v1.analysis import raw_path
from learning.task_completion_bc_v1.data import collect_game
from learning.task_teacher_choice_bc_v1.collect import bounded_turns, pack, selected
from learning.task_teacher_choice_bc_v1.extract import extract
from .games import plans


def teacher_record(record):
    if (not isinstance(record,dict) or record.get('source') != 'teacher'
        or not isinstance(record.get('teacher_label'),str)
        or record.get('teacher_label') != record.get('chosen')):
        raise ValueError('Teacher selection differs from the recorded actor')


def validate_sources(sources,cfg):
    expected = plans(cfg)
    if not isinstance(sources,list) or len(sources) != len(expected):
        raise ValueError('Invalid source count')
    for source, plan in zip(sources, expected):
        if (not isinstance(source,dict) or set(source) != {'game','seed','side','trace','replay_sha256','audit_sha256'}
            or source['game'] != plan['game'] or source['seed'] != plan['seed'] or source['side'] != plan['side']):
            raise ValueError('Source differs from registered plan')
        for name in ('replay_sha256','audit_sha256'):
            value = source[name]
            if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid audit identity')
        trace = source['trace']
        if not isinstance(trace,dict) or set(trace) != {'path','size','sha256'}:
            raise ValueError('Invalid trace identity')
        if trace['path'] != 'traces/'+source['game']+'.jsonl.gz':
            raise ValueError('Trace is not the registered game')


def collect(out,sources,schema,cfg,check):
    """Decode each trace once after the caller binds source and audit identities."""
    validate_sources(sources,cfg)
    limits = cfg['collection_limits']
    meter = Meter(limits['compressed_bytes'],limits['decoded_bytes'])
    groups, rows, checked = [], 0, 0
    for source in sources:
        check()
        game = dict(key=source['game'],side=source['side'],seed=source['seed'])
        path = raw_path(Path(out),source['trace'],'trace')
        stream = bounded_turns(path,source['trace'],meter,check,limits['max_line_bytes'])
        def selected_turns():
            nonlocal rows, checked
            for turn in stream:
                check()
                for ordinal,record in enumerate(turn['records']):
                    teacher_record(record)
                    checked += 1
                    if selected(source['game'],turn['step'],ordinal):
                        item = extract(record,schema,side=source['side'],step=turn['step'],
                                       max_candidates=limits['max_candidates_per_group'])
                        rows += len(item['candidates'])
                        if len(groups)+1 > limits['max_groups'] or rows > limits['max_candidates']:
                            raise ValueError('Sample budget exceeded')
                        groups.append(dict(item,seed=source['seed']))
                yield turn
        try:
            _,unused = collect_game(game,[],selected_turns(),schema)
            if unused:
                raise ValueError('Unexpected episode groups')
        finally:
            stream.close()
    snapshot = meter.snapshot()
    if (snapshot['files_opened'] != 4 or snapshot['files_completed'] != 4
        or snapshot['raw_decode_passes_per_file'] != 1):
        raise ValueError('Incomplete source integrity')
    arrays = pack(groups,limits['max_groups'],limits['max_candidates'])
    return arrays,snapshot,checked
