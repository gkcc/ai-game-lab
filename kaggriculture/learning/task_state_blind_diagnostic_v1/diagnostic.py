"""Private aggregate-only evaluation of fixed historical callback samples."""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from learning.task_multisource_rank_v1 import core as rank, dedup
from learning.task_state_residual_v1 import evaluate
from .sampling import OPPONENTS, ROLES, WINDOWS


@dataclass(frozen=True, slots=True, repr=False)
class PrivateRecord:
    role: str
    opponent: str
    side: int
    window: int
    item: dedup.Item
    student_index: int

    def __repr__(self):
        return '<PrivateBlindRecord>'


def score(records, network, *, guard):
    """Re-score only features and verify the historic baseline choice."""
    if type(records) is not list or len(records) > 192 or not callable(guard):
        raise ValueError('Invalid private diagnostic records')
    groups = []
    for record in records:
        guard()
        if (type(record) is not PrivateRecord or record.role not in ROLES
                or record.opponent not in OPPONENTS or type(record.side) is not int
                or record.side not in (0, 1) or type(record.window) is not int
                or not 0 <= record.window < len(WINDOWS) or record.item.role != record.role
                or type(record.student_index) is not int):
            raise ValueError('Invalid private source coordinates')
        dedup._validate(record.item)
        item = record.item
        scores = network.score(item.shared, item.candidates)
        if (type(scores) is not np.ndarray or scores.dtype != np.dtype('float64')
                or scores.shape != (len(item.candidates),) or not np.isfinite(scores).all()
                or int(np.argmax(scores)) != record.student_index):
            raise ValueError('Historical student choice differs')
        frozen_scores = np.frombuffer(scores.tobytes(), dtype=np.float64)
        groups.append(rank.Group(item.shared, item.candidates, frozen_scores, item.target, 'new'))
    guard()
    return groups


def _duplicates(records, mean, scale, guard):
    raw, normalized = [], []
    for record in records:
        guard()
        left, right = dedup._keys(record.item, mean, scale)
        raw.append(left)
        normalized.append(right)
    def counts(values):
        frequencies = Counter(values)  # bytes equality resolves hash collisions
        return dict(unique=len(frequencies), repeated_groups=sum(n for n in frequencies.values() if n > 1),
                    excess=sum(n - 1 for n in frequencies.values()))
    return dict(raw=counts(raw), normalized=counts(normalized),
                scope='within_selected_ordered_groups_only')


def assess(records, groups, *, mean, scale, params, metadata, guard):
    """Return descriptive aggregate counts; no samples or model data leave."""
    if (type(records) is not list or type(groups) is not list or len(records) != len(groups)
            or len(records) > 192 or not callable(guard)):
        raise ValueError('Invalid private diagnostic population')
    for record, group in zip(records, groups):
        guard()
        if (type(record) is not PrivateRecord or type(group) is not rank.Group
                or record.item.shared is not group.shared or record.item.candidates is not group.candidates
                or record.item.target != group.target or not 0 <= record.student_index < len(group.candidates)
                or int(np.argmax(group.baseline)) != record.student_index):
            raise ValueError('Private scoring population differs')
    def evaluate_by(labels, vocabulary):
        return evaluate.assess(groups, mean, scale, params, metadata, roles=labels,
                               role_labels=vocabulary, guard=guard)
    result = evaluate_by([record.role for record in records], ROLES)
    opponent_alias = dict(market='M01', round2='M02')
    seat_alias = {0: 'S00', 1: 'S01'}
    window_alias = {i: 'W%02d' % i for i in range(len(WINDOWS))}
    opponents = evaluate_by([opponent_alias[record.opponent] for record in records],
                            tuple(opponent_alias.values()))
    seats = evaluate_by([seat_alias[record.side] for record in records], tuple(seat_alias.values()))
    windows = evaluate_by([window_alias[record.window] for record in records], tuple(window_alias.values()))
    def paired(bucket):
        value = dict(bucket)
        four = dict(both_match=bucket['baseline_correct'] - bucket['regressed'],
                    improved=bucket['improved'], regressed=bucket['regressed'],
                    neither_match=bucket['groups'] - bucket['baseline_correct'] - bucket['improved'])
        if (min(four.values()) < 0 or sum(four.values()) != bucket['groups']
                or bucket['residual_correct'] != four['both_match'] + four['improved']):
            raise ValueError('Invalid private paired accounting')
        value['teacher_pair'] = four
        return value
    def flat(value, aliases=None):
        if aliases is None:
            return {key: paired(row['overall']) for key, row in value['by_role'].items()}
        return {str(name): paired(value['by_role'][alias]['overall']) for name, alias in aliases.items()}
    coverage = {role: {opponent: {str(side): dict(groups=0, windows={str(i):0
        for i in range(len(WINDOWS))}) for side in (0, 1)} for opponent in OPPONENTS} for role in ROLES}
    for record in records:
        row = coverage[record.role][record.opponent][str(record.side)]
        row['groups'] += 1
        row['windows'][str(record.window)] += 1
    return dict(overall=paired(result['overall']),
        by_role=flat(result), by_opponent=flat(opponents, opponent_alias), by_side=flat(seats, seat_alias),
        by_window=flat(windows, window_alias), by_target_operation={k: paired(v)
            for k, v in result['by_target_operation'].items()},
        coverage_by_case=coverage,
        single_candidate_groups=sum(len(record.item.candidates) == 1 for record in records),
        duplicates=_duplicates(records, mean, scale, guard),
        independent_evaluation=False, promotion=False, historical_screen='failed')
