"""Compare an already authenticated pair of full replays and task traces.

No file IO, game execution, model loading or identity authentication occurs
here.  Every argument and the returned comparison are private to the caller.
"""
from .consumer import CommonHistoryComparison, OFFICIAL_TURNS, _bytes, _require


def _replay(value, seed):
    _require(type(value) is dict and type(value.get('steps')) is list
             and len(value['steps']) == OFFICIAL_TURNS + 1,
             'Expected exactly 720 replay frames')
    info, cfg = value.get('info'), value.get('configuration')
    _require(type(info) is dict and type(info.get('seed')) is int and info['seed'] == seed
             and type(cfg) is dict, 'Replay seed or configuration differs')
    _bytes(cfg)
    for step, frame in enumerate(value['steps']):
        _require(type(frame) is list and len(frame) == 2
                 and all(type(state) is dict and type(state.get('observation')) is dict for state in frame)
                 and type(frame[0]['observation'].get('step')) is int
                 and frame[0]['observation']['step'] == step, 'Malformed or unordered replay frame')
        if step:
            _require(all('action' in state for state in frame), 'Missing replay action')
    _require([state.get('status') for state in value['steps'][-1]] == ['DONE', 'DONE'],
             'Unfinished replay')
    return value['steps']


def _trace_line(iterator, step):
    try:
        line = next(iterator)
    except StopIteration:
        raise ValueError('Missing trace line') from None
    _require(type(line) is dict and type(line.get('step')) is int and line['step'] == step
             and type(line.get('records')) is list
             and all(type(line.get(k)) is int and line[k] == 0
                     for k in ('load_errors', 'inference_errors', 'fallbacks')),
             'Malformed, reordered or failing trace line')
    return line


def _trace_end(iterator):
    # Advancing to EOF lets a caller's streaming decoder check its digest.
    try:
        next(iterator)
    except StopIteration:
        return
    raise ValueError('Extra trace line')


def compare_replays(reference_replay, candidate_replay, reference_trace, candidate_trace, *,
                    reference, candidate, seed, seed_alias, opponent, side):
    """Return one private comparison row for ``panel.paired(comparisons=...)``.

    The two inputs must already have been separately authenticated and audited.
    Trace iterables are consumed in lockstep, through EOF even after the first
    divergence.  Callers must bind and bound each stream, verify its digest by
    EOF, and close it after any exception. Each action passed to the comparison
    comes from the next replay frame, never the current frame. ``seed_alias``
    and arm/opponent labels still require external provenance.
    """
    _require(type(reference) is str and bool(reference) and type(candidate) is str
             and bool(candidate) and reference != candidate, 'Invalid policy roles')
    _require(type(seed) is int and 0 <= seed < 2**32 and type(seed_alias) is str and bool(seed_alias)
             and type(opponent) is str and bool(opponent) and type(side) is int and side in (0, 1),
             'Invalid comparison coordinate')
    ref = _replay(reference_replay, seed)
    new = _replay(candidate_replay, seed)
    ref_trace, new_trace = iter(reference_trace), iter(candidate_trace)
    _require(_bytes(reference_replay['configuration']) == _bytes(candidate_replay['configuration']),
             'Replay configurations differ')
    comparison = CommonHistoryComparison(side)
    for step in range(OFFICIAL_TURNS):
        ref_line, new_line = _trace_line(ref_trace, step), _trace_line(new_trace, step)
        if not comparison.stopped:
            ref_actions = [state['action'] for state in ref[step + 1]]
            new_actions = [state['action'] for state in new[step + 1]]
            comparison.observe(step, ref[step], new[step], ref_line, new_line,
                               ref_actions, new_actions)
    _trace_end(ref_trace)
    _trace_end(new_trace)
    result = comparison.result()
    _require(result['comparison_complete'], 'Comparison did not complete')
    return dict(reference=reference, candidate=candidate, seed=seed, seed_alias=seed_alias,
                opponent=opponent, side=side, result=result)
