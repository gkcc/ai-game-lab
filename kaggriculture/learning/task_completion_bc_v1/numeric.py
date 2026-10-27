"""Private numeric cache and preregistered exploratory offline gate."""
import numpy as np
from learning.task_choice_v1.features import OPS

_KEYS = {'shared', 'candidates', 'offsets', 'chosen', 'ops', 'seed', 'status'}
_DTYPES = {'shared': np.dtype('float32'), 'candidates': np.dtype('float32'),
           'offsets': np.dtype('int64'), 'chosen': np.dtype('int64'),
           'ops': np.dtype('uint8'), 'seed': np.dtype('uint32'), 'status': np.dtype('uint8')}
_STATUSES = ('completed_observed', 'unknown')


def _limits(max_groups, max_candidates):
    for x in (max_groups, max_candidates):
        if type(x) is not int or x < 0:
            raise ValueError('limits must be nonnegative integers')


def pack(groups, max_groups, max_candidates):
    _limits(max_groups, max_candidates)
    if not isinstance(groups, (list, tuple)) or len(groups) > max_groups:
        raise ValueError('invalid group count')
    shared, candidates, chosen, ops, seeds, statuses, offsets = [], [], [], [], [], [], [0]
    for g in groups:
        if not isinstance(g, dict) or set(g) != {'shared', 'candidates', 'chosen', 'candidate_ops', 'seed', 'status'}:
            raise ValueError('invalid group fields')
        s, c = np.asarray(g['shared']), np.asarray(g['candidates'])
        if s.dtype != np.float32 or c.dtype != np.float32 or s.shape != (275,) or c.ndim != 2 or c.shape[1] != 98 or len(c) == 0:
            raise ValueError('invalid feature shape or dtype')
        if not np.isfinite(s).all() or not np.isfinite(c).all() or offsets[-1] + len(c) > max_candidates:
            raise ValueError('invalid features or candidate count')
        ch, seed = g['chosen'], g['seed']
        if isinstance(ch, (bool, np.bool_)) or not isinstance(ch, (int, np.integer)) or not 0 <= ch < len(c):
            raise ValueError('invalid chosen')
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 0xffffffff:
            raise ValueError('invalid seed')
        op = g['candidate_ops']
        if not isinstance(op, (list, tuple, np.ndarray)) or len(op) != len(c) or any(not isinstance(x, str) or x not in OPS for x in op):
            raise ValueError('invalid operations')
        if g['status'] not in _STATUSES:
            raise ValueError('invalid status')
        shared.append(s); candidates.append(c); chosen.append(ch); ops.extend(OPS.index(x) for x in op)
        seeds.append(seed); statuses.append(_STATUSES.index(g['status'])); offsets.append(offsets[-1] + len(c))
    return dict(shared=np.asarray(shared, dtype=np.float32).reshape(-1, 275),
                candidates=np.concatenate(candidates) if candidates else np.empty((0, 98), np.float32),
                offsets=np.asarray(offsets, np.int64), chosen=np.asarray(chosen, np.int64),
                ops=np.asarray(ops, np.uint8), seed=np.asarray(seeds, np.uint32), status=np.asarray(statuses, np.uint8))


def unpack(arrays, max_groups, max_candidates):
    _limits(max_groups, max_candidates)
    if not isinstance(arrays, dict) or set(arrays) != _KEYS:
        raise ValueError('invalid array fields')
    if any(not isinstance(arrays[k], np.ndarray) or arrays[k].dtype != d for k, d in _DTYPES.items()):
        raise ValueError('invalid array dtype')
    s, c, o = arrays['shared'], arrays['candidates'], arrays['offsets']
    if s.ndim != 2 or s.shape[1] != 275 or len(s) > max_groups or c.ndim != 2 or c.shape[1] != 98 or len(c) > max_candidates:
        raise ValueError('invalid dimensions or limits')
    n = len(s)
    if o.shape != (n + 1,) or o[0] != 0 or o[-1] != len(c) or np.any(o[1:] <= o[:-1]):
        raise ValueError('invalid offsets')
    if any(arrays[k].shape != (n,) for k in ('chosen', 'seed', 'status')) or arrays['ops'].shape != (len(c),):
        raise ValueError('invalid metadata shape')
    if not np.isfinite(s).all() or not np.isfinite(c).all() or np.any(arrays['ops'] >= len(OPS)) or np.any(arrays['status'] >= len(_STATUSES)):
        raise ValueError('invalid feature or metadata value')
    if np.any(arrays['chosen'] < 0) or np.any(arrays['chosen'] >= np.diff(o)):
        raise ValueError('invalid chosen')
    return [dict(shared=s[i].copy(), candidates=c[o[i]:o[i+1]].copy(), chosen=int(arrays['chosen'][i]),
                 candidate_ops=[OPS[int(x)] for x in arrays['ops'][o[i]:o[i+1]]], seed=int(arrays['seed'][i]),
                 status=_STATUSES[int(arrays['status'][i])]) for i in range(n)]


def write_new(path, arrays):
    # Validate before touching the destination; exclusive creation avoids overwriting.
    unpack(arrays, len(arrays['shared']), len(arrays['candidates']))
    with open(path, 'xb') as f:
        np.savez(f, **arrays)


def read(path, max_groups, max_candidates):
    _limits(max_groups, max_candidates)
    # Bound decompression based on the declared NPY header sizes before loading.
    import zipfile
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if len(infos) != len(_KEYS) or {i.filename for i in infos} != {k + '.npy' for k in _KEYS}:
            raise ValueError('invalid archive')
        ceiling = max_groups * (275 * 4 + 8 + 8 + 4 + 1) + max_candidates * (98 * 4 + 1) + 8 + 4096 * len(_KEYS)
        if sum(i.file_size for i in infos) > ceiling:
            raise ValueError('archive exceeds limits')
        if hasattr(path, 'seek'): path.seek(0)
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {k: loaded[k] for k in loaded.files}
    unpack(arrays, max_groups, max_candidates)
    return arrays


def exploratory_gate(metrics, expected_seeds):
    """Fixed preregistered offline screening, not significance or deployment evidence."""
    failures = []
    def metric(group, field):
        try:
            v = group[field]
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v):
                raise ValueError
            return float(v)
        except (KeyError, TypeError, ValueError):
            failures.append('missing or nonfinite ' + field)
            return None
    def compare(group, model, baseline, threshold, strict=False):
        a, b = metric(group, model), metric(group, baseline)
        if a is not None and b is not None and ((a - b <= threshold) if strict else (a - b < threshold - 1e-12)):
            failures.append(model + ' threshold failed')
    if not isinstance(metrics, dict) or not isinstance(expected_seeds, (list, tuple, set)) or not expected_seeds or len({str(s) for s in expected_seeds}) != len(expected_seeds):
        return {'passed': False, 'failures': ['missing metrics or expected seeds']}
    def inspect_values(value):
        if isinstance(value, dict):
            for item in value.values(): inspect_values(item)
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            failures.append('nonfinite metric')
    inspect_values(metrics)
    overall = metrics.get('overall', {})
    count = metric(overall, 'count')
    if count is not None and (count < 1 or not count.is_integer()): failures.append('overall empty or invalid count')
    compare(overall, 'model_top1', 'baseline_top1', 0.05)
    compare(overall, 'baseline_ce', 'model_ce', 0)
    by_seed = metrics.get('by_seed', {})
    for seed in expected_seeds:
        group = by_seed.get(str(seed), {}) if isinstance(by_seed, dict) else {}
        count = metric(group, 'count')
        if count is not None and (count < 1 or not count.is_integer()): failures.append('seed empty or invalid count')
        compare(group, 'model_top1', 'baseline_top1', 0, True)
    for name in ('multi_candidate', 'cross_operation'):
        group = metrics.get(name, {})
        count = metric(group, 'count')
        if count is not None and (count < 1 or not count.is_integer()): failures.append(name + ' empty or invalid count')
        compare(group, 'model_top1', 'baseline_top1', 0)
    by_op = metrics.get('by_operation', {})
    gains = []
    for op in OPS:
        group = by_op.get(op, {}) if isinstance(by_op, dict) else {}
        count = metric(group, 'count')
        if count is None: continue
        if count < 0 or not count.is_integer(): failures.append(op + ' invalid count'); continue
        if count >= 20:
            a, b = metric(group, 'model_operation_accuracy'), metric(group, 'baseline_operation_accuracy')
            if a is not None and b is not None: gains.append(a - b)
    if len(gains) < 3: failures.append('fewer than three eligible operations')
    elif np.mean(gains) < 0.05 - 1e-12: failures.append('operation macro gain threshold failed')
    return {'passed': not failures, 'failures': failures}
