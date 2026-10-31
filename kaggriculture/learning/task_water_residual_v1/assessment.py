"""Private, group-level preparation and consumed-development assessment.

No files, devices, model updates, games, or unrevealed seeds are accessed here.
The caller authenticates the model, source groups and runtime budgets.
"""
import numpy as np

from learning.task_choice_v1.features import OPS
from . import policy

TRAIN_ROLES = ('A01', 'A02')
DEV_ROLES = ('A07', 'A08')


def _scored(group, network, metadata):
    schema = policy._schema(metadata)
    if policy._schema(network.metadata) != schema:
        raise ValueError('Task model schema differs')
    raw = group.features
    if (type(raw) is not np.ndarray or raw.dtype != np.dtype('float32') or raw.ndim != 2
            or raw.shape[1] != policy.DIM or not 1 <= len(raw) <= 256
            or not np.isfinite(raw).all()
            or not np.array_equal(raw[:, :policy.SHARED_DIM],
                                  np.broadcast_to(raw[:1, :policy.SHARED_DIM], (len(raw), policy.SHARED_DIM)))
            or type(group.target_index) is not int or not 0 <= group.target_index < len(raw)):
        raise ValueError('Task group input differs')
    ops = policy._operations(raw, schema)
    target_op = OPS[int(np.argmax(ops[group.target_index]))]
    if target_op != group.op:
        raise ValueError('Task group target operation differs')
    scores = network.score(raw[0, :policy.SHARED_DIM], raw[:, policy.SHARED_DIM:])
    if (type(scores) is not np.ndarray or scores.dtype != np.dtype('float64')
            or scores.shape != (len(raw),) or not np.isfinite(scores).all()):
        raise ValueError('Task model scores differ')
    return raw, ops[:, OPS.index('WATER')] == 1, scores


def prepare_training(groups, network, metadata, *, guard):
    """Keep WATER-teacher groups where the frozen model routes to WATER.

    Only this returned WATER subset enters the optimizer. The target is
    remapped to its index in that subset; role is used only to balance batches.
    """
    if type(groups) not in (tuple, list) or not callable(guard) or len(groups) > 384:
        raise ValueError('Invalid training groups')
    selected, counts, float32_ties = [], {role: 0 for role in TRAIN_ROLES}, 0
    for group in groups:
        guard()
        if group.role not in TRAIN_ROLES:
            raise ValueError('Training role differs')
        raw, water, scores = _scored(group, network, metadata)
        original = int(np.argmax(scores))
        if group.op != 'WATER' or not water[original] or np.count_nonzero(water) < 2:
            continue
        indices = np.flatnonzero(water)
        target = int(np.flatnonzero(indices == group.target_index)[0])
        local_scores = scores[indices].astype(np.float32)
        if not np.isfinite(local_scores).all():
            raise ValueError('Training score cast is not finite')
        float32_ties += int(int(np.argmax(local_scores)) != int(np.argmax(scores[indices])))
        selected.append(dict(role=TRAIN_ROLES.index(group.role), features=raw[indices].copy(),
                             baseline_logits=local_scores, teacher_index=target))
        counts[group.role] += 1
    return selected, dict(input_groups=len(groups), selected_groups=len(selected),
                          selected_by_role=counts, float32_baseline_argmax_differences=float32_ties)


def _nll(scores, target):
    maximum = float(np.max(scores))
    value = (maximum - float(scores[target])) + float(np.log(np.exp(scores - maximum).sum()))
    if not np.isfinite(value):
        raise ValueError('Invalid conditional teacher likelihood')
    return value


def _summary(rows, roles):
    eligible = [row for row in rows if row['eligible']]
    by_role = {}
    for role in roles:
        local = [row for row in eligible if row['role'] == role]
        by_role[role] = dict(groups=len(local), baseline_correct=sum(r['baseline_correct'] for r in local),
            residual_correct=sum(r['residual_correct'] for r in local),
            baseline_nll=(sum(r['baseline_nll'] for r in local) / len(local) if local else None),
            residual_nll=(sum(r['residual_nll'] for r in local) / len(local) if local else None))
    return dict(groups=len(rows), eligible_groups=len(eligible), roles=by_role,
        baseline_correct=sum(r['baseline_correct'] for r in eligible),
        residual_correct=sum(r['residual_correct'] for r in eligible),
        improved=sum(not r['baseline_correct'] and r['residual_correct'] for r in eligible),
        regressed=sum(r['baseline_correct'] and not r['residual_correct'] for r in eligible),
        changed=sum(r['changed'] for r in eligible),
        operation_flips=sum(r['operation_flip'] for r in rows))


def assess_development(groups, network, weights, metadata, *, guard):
    """Compare one frozen candidate on consumed dev groups, never a match.

    Returns aggregate metrics and private per-group booleans/NLL, not features,
    identifiers or logits. Calling code decides which aggregate can be shared.
    """
    if (type(groups) not in (tuple, list) or len(groups) > 384 or not callable(guard)
            or type(weights) is not np.ndarray or weights.dtype != np.dtype('float32')
            or weights.shape != (policy.DIM,) or not np.isfinite(weights).all()):
        raise ValueError('Invalid development groups or weights')
    schema = policy._schema(metadata)
    mean, scale = policy._normalization(network.mean, network.scale, schema)
    rows = []
    for group in groups:
        guard()
        if (group.role not in DEV_ROLES or group.opponent not in ('market', 'round2')
                or type(group.side) is not int or group.side not in (0, 1)):
            raise ValueError('Development coordinate differs')
        raw, water, scores = _scored(group, network, metadata)
        base = int(np.argmax(scores))
        candidate = policy.select(raw, scores, mean, scale, weights, metadata)
        operations = policy._operations(raw, schema)
        op_flip = bool(not np.array_equal(operations[base], operations[candidate]))
        eligible = bool(group.op == 'WATER' and water[base] and np.count_nonzero(water) >= 2)
        row = dict(role=group.role, opponent=group.opponent, side=group.side, eligible=eligible,
                   changed=bool(candidate != base), operation_flip=op_flip,
                   baseline_operation=OPS[int(np.argmax(operations[base]))],
                   water_candidates=int(np.count_nonzero(water)),
                   baseline_correct=bool(base == group.target_index),
                   residual_correct=bool(candidate == group.target_index),
                   baseline_nll=None, residual_nll=None)
        if eligible:
            indices = np.flatnonzero(water)
            target = int(np.flatnonzero(indices == group.target_index)[0])
            normalized = ((raw[indices].astype(np.float64) - mean) / scale).astype(np.float32)
            residual = normalized @ weights
            if not np.isfinite(normalized).all() or not np.isfinite(residual).all():
                raise ValueError('Invalid residual scores')
            row['baseline_nll'] = _nll(scores[indices], target)
            row['residual_nll'] = _nll(scores[indices] + residual.astype(np.float64), target)
        rows.append(row)
    aggregate = _summary(rows, DEV_ROLES)
    aggregate['opponents'] = {name: _summary([r for r in rows if r['opponent'] == name], DEV_ROLES)
                              for name in ('market', 'round2')}
    aggregate['sides'] = {str(side): _summary([r for r in rows if r['side'] == side], DEV_ROLES)
                          for side in (0, 1)}
    roles = aggregate['roles']
    enough = all(roles[role]['groups'] > 0 for role in DEV_ROLES)
    observed_role_drop = any(roles[r]['groups'] and roles[r]['residual_correct'] < roles[r]['baseline_correct']
                             for r in DEV_ROLES)
    role_guard = False if observed_role_drop else True if enough else None
    nll_guard = (sum(roles[r]['residual_nll'] for r in DEV_ROLES) / len(DEV_ROLES)
                 <= sum(roles[r]['baseline_nll'] for r in DEV_ROLES) / len(DEV_ROLES) + 1e-7) if enough else None
    checks = dict(each_role_has_eligible=enough, each_role_no_accuracy_drop=role_guard,
                  two_net_improvements=(aggregate['improved'] - aggregate['regressed'] >= 2) if enough else None,
                  equal_role_nll_no_increase=nll_guard,
                  at_least_one_change=True if aggregate['changed'] else False if enough else None,
                  no_operation_flips=aggregate['operation_flips'] == 0)
    known_failure = aggregate['operation_flips'] > 0 or observed_role_drop
    aggregate['checks'] = checks
    aggregate['screen'] = ('failed' if known_failure else 'unknown' if not enough
                           else 'passed' if all(checks.values()) else 'failed')
    return aggregate, rows
