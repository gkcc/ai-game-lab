"""Fail-closed NumPy task candidate scorer; no teacher fallback or game imports."""
import hashlib
import json
from pathlib import Path

import numpy as np

from learning.task_choice_v1.features import SCHEMA_VERSION, OPS, schema_hash

DIMENSIONS = (373, 128, 64, 1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def continuous_mask(names):
    """Preserve one-hot/presence/binary semantics; normalize other train columns."""
    binary_suffixes = ('.present', '.watered_today', '.fed_today', '.cared_today',
                       '.fertilizer_available', '.pending_care_bonus')
    return np.asarray([not (name == 'planning.final_day' or name.endswith(binary_suffixes)
                            or name.startswith('task.op.')
                            or any(part in name for part in ('.kind.', '.crop.', '.animal.'))
                            or '.below_func.' in name or '.above_func.' in name)
                       for name in names], dtype=bool)


def numpy_logits(params, normalized):
    x = np.asarray(normalized, dtype=np.float64)
    for layer in range(3):
        x = x @ np.asarray(params['w'+str(layer)], dtype=np.float64) + np.asarray(params['b'+str(layer)], dtype=np.float64)
        if layer < 2:
            x = np.maximum(x, 0)
    result = x[..., 0]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite task model logits')
    return result


class TaskNetwork:
    """Callable TaskAgent task_selector; first canonical candidate wins exact ties."""
    def __init__(self, path, expected_sha):
        if not expected_sha or sha(path) != expected_sha:
            raise ValueError('Missing or mismatched task checkpoint identity')
        self.sha = expected_sha
        with np.load(path, allow_pickle=False) as z:
            self.metadata = json.loads(str(z['metadata']))
            self.mean, self.scale = z['mean'].astype(np.float64), z['scale'].astype(np.float64)
            self.params = {k: z[k].astype(np.float64) for k in ('w0','b0','w1','b1','w2','b2')}
        m = self.metadata
        if (m['schema_version'] != SCHEMA_VERSION or m['ops'] != list(OPS)
                or m['dimensions'] != list(DIMENSIONS)
                or len(m['shared_feature_names']) != 275 or len(m['feature_names']) != 98
                or m['schema_sha256'] != schema_hash(m['feature_names'], m['shared_feature_names'])):
            raise ValueError('Incompatible task schema')
        if self.mean.shape != (373,) or self.scale.shape != (373,):
            raise ValueError('Invalid normalization shape')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or not (self.scale >= 1).all():
            raise ValueError('Invalid normalization values')
        mask = continuous_mask(m['shared_feature_names'] + m['feature_names'])
        if not np.all(self.mean[~mask] == 0) or not np.all(self.scale[~mask] == 1):
            raise ValueError('Binary inputs must remain unnormalized')
        for i in range(3):
            w, b = self.params['w'+str(i)], self.params['b'+str(i)]
            if w.shape != (DIMENSIONS[i], DIMENSIONS[i+1]) or b.shape != (DIMENSIONS[i+1],):
                raise ValueError('Invalid task layer shape')
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('Nonfinite task weights')
        self.last = None

    def score(self, shared, candidates):
        shared, candidates = np.asarray(shared, dtype=np.float64), np.asarray(candidates, dtype=np.float64)
        if shared.shape != (275,) or candidates.ndim != 2 or candidates.shape[1] != 98 or not len(candidates):
            raise ValueError('Invalid task scoring input shape')
        x = np.concatenate((np.broadcast_to(shared, (len(candidates), 275)), candidates), axis=1)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite task input')
        return numpy_logits(self.params, (x-self.mean)/self.scale)

    def __call__(self, inputs):
        self.last = None
        if (inputs['schema_version'] != SCHEMA_VERSION or inputs['schema_sha256'] != self.metadata['schema_sha256']
                or inputs['shared_feature_names'] != self.metadata['shared_feature_names']
                or inputs['feature_names'] != self.metadata['feature_names']):
            raise ValueError('Runtime task schema mismatch')
        candidates = inputs['candidates']
        ids = [c['id'] for c in candidates]
        if any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
            raise ValueError('Invalid candidate identity')
        logits = self.score(inputs['shared_x'], [c['x'] for c in candidates])
        selected = int(np.argmax(logits))
        self.last = dict(checkpoint_sha256=self.sha, chosen=ids[selected], logits=logits.tolist(), fallbacks=0)
        return ids[selected]
