"""Analytic softmax derivatives in float64. Does not change model parameters."""
import hashlib
import numpy as np


def objective(h, y, theta, weight=None):
    h, theta = np.asarray(h, dtype=np.float64), np.asarray(theta, dtype=np.float64)
    weight = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=np.float64)
    if not len(y) or weight.shape != (len(y),) or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError('Invalid objective weights')
    z = h @ theta
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z); p = exp / exp.sum(axis=1, keepdims=True)
    loss = np.dot(weight, np.log(exp.sum(axis=1)) - z[np.arange(len(y)), y]) / weight.sum()
    residual = p.copy(); residual[np.arange(len(y)), y] -= 1
    grad = h.T @ (residual * (weight / weight.sum())[:, None])
    assert np.isfinite(loss) and np.isfinite(grad).all()
    return float(loss), grad, p


def direction_stats(h, weight, p, grad, direction):
    projected = h @ direction
    mean = (p * projected).sum(axis=1, keepdims=True)
    variance = (p * (projected - mean)**2).sum(axis=1)
    return dict(derivative=float(np.sum(grad * direction)),
                curvature=float(np.dot(weight, variance) / np.sum(weight)))


def first_adam(grad, learning_rate, epsilon):
    """Ideal bias-corrected first step, zero moments, eps_root=0; no update."""
    return -learning_rate * grad / (np.abs(grad) + epsilon)


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a*b) / denominator) if denominator else None


def feature_hash(x):
    return hashlib.sha256(np.asarray(x, dtype='<f4').tobytes()).hexdigest()


def collision_summary(hidden, labels):
    groups = {}
    for row, label in zip(hidden, labels):
        row = np.asarray(row, dtype='<f8').copy(); row[row == 0] = 0
        if not np.isfinite(row).all(): raise ValueError('Non-finite hidden representation')
        groups.setdefault(row.tobytes(), set()).add(int(label))
    return dict(unique_hidden=len(groups), different_label_groups=sum(len(v)>1 for v in groups.values()))


def nearest(query, pool, signatures, mask):
    ids = np.flatnonzero(mask)
    if not len(ids): return None
    distance = np.linalg.norm(pool[ids] - query, axis=1)
    # Exact numeric distance, then lexicographic feature SHA; no fitted tolerance.
    order = np.lexsort((np.asarray(signatures)[ids], distance))
    return int(ids[order[0]]), float(distance[order[0]])
