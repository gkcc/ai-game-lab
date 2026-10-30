"""Coarse data readiness from private group-level preflight outputs.

Never include source identities, raw features, group masks or exact small counts
in the returned projection. These thresholds are developmental, not anonymity
or evidence of model generalization.
"""
import numpy as np
from learning.task_choice_v1.features import OPS


def support(groups, clusters):
    return 'supported' if groups >= 20 and clusters >= 2 else 'unknown'


def primary_support(stats, role):
    counts = stats['qualifying_per_cluster_sorted']
    total = stats['qualifying_plant_groups']
    if role == 'train':
        return 'supported' if total >= 20 and sum(x > 0 for x in counts) >= 2 else 'unknown'
    return 'supported' if total >= 20 and all(x >= 5 for x in counts) else 'unknown'


def unmatched_coverage(views, masks, expected_clusters):
    """Count privately within cluster; publish only fixed sufficiency flags."""
    operations = {op: {'groups': 0, 'clusters': set()} for op in OPS}
    qualifying = {int(cluster): 0 for cluster in expected_clusters}
    for view, group_masks in zip(views, masks, strict=True):
        unseen = ~group_masks['seen_in_train']
        labels = view.ops[view.offsets[:-1] + view.labels]
        for code, op in enumerate(OPS):
            chosen = unseen & (labels == code)
            operations[op]['groups'] += int(np.count_nonzero(chosen))
            operations[op]['clusters'].update(int(x) for x in np.unique(view.private_clusters[chosen]))
        for cluster in qualifying:
            qualifying[cluster] += int(np.count_nonzero(unseen & group_masks['qualifying_plant']
                                                        & (view.private_clusters == cluster)))
    primary = sum(qualifying.values()) >= 20 and all(n >= 5 for n in qualifying.values())
    return {'qualifying_plant': 'supported' if primary else 'unknown',
            'operations': {op: support(x['groups'], len(x['clusters'])) for op, x in operations.items()}}


def coverage_projection(item, role):
    return {'qualifying_plant': primary_support(item, role),
            'operations': {op: support(item['operations'][op]['groups'],
                                       item['operations'][op]['clusters_with_groups']) for op in OPS}}


def project(private, roles, expected_roles):
    result = {'coverage': {}, 'unmatched_coverage': {}, 'conditional_sources': {},
              'overlap': {}, 'label_conflict': 'none'}
    for role in ('train', 'teacher_dev', 'conditional_dev'):
        item = private.coverage[role]['aggregate']
        result['coverage'][role] = coverage_projection(item, role)
    for role in ('teacher_dev', 'conditional_dev'):
        masks = private.masks[role]
        total_seen = sum(int(np.count_nonzero(group['seen_in_train'])) for group in masks)
        any_conflict = any(bool(np.any(group['label_conflict'])) for group in masks)
        result['overlap'][role] = 'at_least_20' if total_seen >= 20 else 'below_20'
        result['unmatched_coverage'][role] = unmatched_coverage(roles[role], masks, expected_roles[role])
        if any_conflict:
            result['label_conflict'] = 'detected'
    views = roles['conditional_dev']
    masks = private.masks['conditional_dev']
    sources = private.coverage['conditional_dev']['sources']
    if not (len(views) == len(masks) == len(sources) == 2):
        raise ValueError('Conditional source count differs')
    for index, name in enumerate(('old', 'new')):
        clusters = np.unique(views[index].private_clusters)
        if len(clusters) != 2:
            raise ValueError('Conditional source cluster count differs')
        result['conditional_sources'][name] = {
            'coverage': coverage_projection(sources[index], 'conditional_dev'),
            'unmatched_coverage': unmatched_coverage([views[index]], [masks[index]], clusters),
        }
    result['all_groups_primary_ready'] = (result['label_conflict'] == 'none' and all(
        result['coverage'][role]['qualifying_plant'] == 'supported'
        for role in ('train', 'teacher_dev', 'conditional_dev')) and all(
        result['conditional_sources'][name]['coverage']['qualifying_plant'] == 'supported'
        for name in ('old', 'new')))
    result['unmatched_primary_ready'] = (result['all_groups_primary_ready'] and all(
        result['unmatched_coverage'][role]['qualifying_plant'] == 'supported'
        for role in ('teacher_dev', 'conditional_dev')) and all(
        result['conditional_sources'][name]['unmatched_coverage']['qualifying_plant'] == 'supported'
        for name in ('old', 'new')))
    return result
