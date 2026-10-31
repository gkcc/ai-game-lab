"""Private bounded screening of authenticated task disagreements.

No game, model, seed or file is loaded here. The caller must authenticate the
source and pass its still-open game, resource guard, comparer and official
single-step executor. Candidate IDs and source state must remain local.
"""
from dataclasses import dataclass

from decision.api import observe
from decision.audit_market_teacher import observation
from learning.task_bc_error_diagnostics_v1.shadow import ordered_hash
from learning.task_request_gate_v1.policy import same_request
from learning.task_student_dagger_collection_v1 import sampling


ROLES = ('A05', 'A06')
SLOTS = {'A05': 'T01', 'A06': 'T02'}
OPPONENTS = ('market', 'round2')
WINDOWS = sampling.WINDOWS
SELECTED_PER_WINDOW = 2
MAX_CASE_COMPARISONS = len(WINDOWS) * SELECTED_PER_WINDOW
MAX_CANDIDATES_PER_WINDOW = sampling.PER_WINDOW
MAX_CANDIDATE_WIDTH = 256
MAX_COMPARISONS = 96
MAX_OFFICIAL_TRANSITIONS = 192
STATUSES = ('empty', 'unexamined_after_hit', 'request_same', 'official_request_same', 'unknown',
            'official_next_frame_no_difference_detected', 'official_next_frame_different')


def _require(value):
    if not value:
        raise ValueError('Invalid private counterfactual source')


@dataclass(frozen=True, slots=True, repr=False)
class PrivateCase:
    """Anonymous coverage plus a locally sealed candidate, if one exists."""
    coverage: dict
    candidate: object
    plan: list

    def __repr__(self):
        return '<PrivateCounterfactualCase>'


class Meter:
    """Caller-owned counters survive errors; engine internals remain unknown."""
    def __init__(self):
        self.comparisons_reserved = self.comparisons_begun = self.comparisons_completed = 0
        self.controller_turns_reserved = 0
        self.official_transitions_reserved = 0
        self.official_helpers_begun = self.official_helpers_completed = 0

    def reserve_comparison(self):
        _require(self.comparisons_reserved < MAX_COMPARISONS)
        self.comparisons_reserved += 1
        self.controller_turns_reserved += 2
        self.comparisons_begun += 1

    def complete_comparison(self):
        _require(self.comparisons_completed < self.comparisons_begun)
        self.comparisons_completed += 1

    def reserve_official(self):
        _require(self.official_transitions_reserved + 2 <= MAX_OFFICIAL_TRANSITIONS)
        self.official_transitions_reserved += 2
        self.official_helpers_begun += 1

    def complete_official(self):
        _require(self.official_helpers_completed < self.official_helpers_begun)
        self.official_helpers_completed += 1

    def snapshot(self):
        return dict(comparisons_reserved=self.comparisons_reserved,
            comparisons_begun=self.comparisons_begun,comparisons_completed=self.comparisons_completed,
            controller_turns_reserved=self.controller_turns_reserved,
            controller_turns_actual='unknown',
            official_transitions_reserved=self.official_transitions_reserved,
            official_helpers_begun=self.official_helpers_begun,
            official_helpers_completed=self.official_helpers_completed,
            official_transitions_confirmed_completed=2*self.official_helpers_completed,
            incomplete_helper_step_progress=('unknown' if self.official_helpers_begun
                                             != self.official_helpers_completed else 'none'))

    def __repr__(self):
        return '<PrivateCounterfactualMeter>'
def _select(windows, *, slot, opponent, side):
    """Take the first two already-ranked distinct steps per fixed window."""
    _require(type(windows) is list and len(windows) == len(WINDOWS))
    selected, outside = [], []
    for window, entries in enumerate(windows):
        _require(type(entries) is list and len(entries) <= MAX_CANDIDATES_PER_WINDOW)
        seen, ranks = set(), []
        for item in entries:
            _require(type(item) is dict and set(item) == {
                'step','ordinal','third_candidate','document','previous_document'})
            step, ordinal = item['step'], item['ordinal']
            _require(type(step) is int and WINDOWS[window][0] <= step <= WINDOWS[window][1]
                     and type(ordinal) is int and 0 <= ordinal < 32 and step not in seen)
            seen.add(step)
            ranks.append((sampling.coordinate_rank(slot,opponent,side,step,ordinal), step, ordinal))
        _require(ranks == sorted(ranks))
        selected.append(entries[:SELECTED_PER_WINDOW])
        outside.append(max(0, len(entries)-SELECTED_PER_WINDOW))
    return selected, outside


def run_case(game, *, role, opponent, side, guard, compare_turn, compare_step, meter):
    """Stop at the first v2-canonical/Python non-action next-frame difference."""
    _require(role in ROLES and opponent in OPPONENTS and type(side) is int and side in (0,1)
             and callable(guard) and callable(compare_turn) and callable(compare_step) and type(meter) is Meter)
    slot = SLOTS[role]
    identity = game.identity
    _require(type(identity) is dict and all(identity.get(key) == value for key,value in (
        ('source_role',role),('batch',2),('role',slot),('opponent',opponent),('side',side))))
    guard()
    accounting, windows = game.scan_candidates(guard=guard)
    _require(type(accounting) is dict and accounting.get('lines') == 719)
    selected, outside = _select(windows, slot=slot, opponent=opponent, side=side)
    # Fix all selected coordinates before any outcome or model comparison.
    plan = [[dict(step=entry['step'],ordinal=entry['ordinal']) for entry in entries]
            for entries in selected]
    plan_sha = ordered_hash(plan)
    states = [['empty'] * SELECTED_PER_WINDOW for _ in WINDOWS]
    comparisons = transitions = 0
    candidate = None
    for window, entries in enumerate(selected):
        for index, entry in enumerate(entries):
            guard()
            if candidate is not None:
                states[window][index] = 'unexamined_after_hit'
                continue
            step, ordinal = entry['step'], entry['ordinal']
            _require(ordered_hash(plan) == plan_sha
                     and plan[window][index] == dict(step=step,ordinal=ordinal))
            doc = entry['document']
            _require(type(doc) is dict and doc.get('step') == step and type(doc.get('records')) is list
                     and ordinal < len(doc['records']))
            record = doc['records'][ordinal]
            _require(type(record) is dict and record.get('source') == 'callback'
                and type(record.get('chosen')) is str and type(record.get('teacher_label')) is str
                and record['chosen'] != record['teacher_label'] and type(record.get('inputs')) is dict)
            inputs = record['inputs']
            _require(type(inputs.get('candidates')) is list and 1 <= len(inputs['candidates']) <= MAX_CANDIDATE_WIDTH)
            _require(comparisons < MAX_CASE_COMPARISONS)
            view = observe(observation(game.replay['steps'][step],side), game.replay['configuration'])
            _require('seed' not in view.configuration and 'info' not in view.observation)
            original = game.replay['steps'][step+1][side]['action']
            comparisons += 1
            meter.reserve_comparison()
            result = compare_turn(doc,view.observation,view.configuration,original,ordinal,
                task_sha256=game.task_sha256,crop_sha256=game.crop_sha256,game_limits=game.game_limits,
                agent_factory=game.make_agent,guard=guard,previous_document=entry['previous_document'],
                alternative='teacher')
            _require(result.status in ('same','different','unknown'))
            if result.status == 'unknown':
                meter.complete_comparison()
                states[window][index] = 'unknown'
                continue
            _require(type(result.checks) is dict and type(result.checks.get('request_equal')) is bool)
            if result.status == 'same':
                _require(result.checks['request_equal'] is True)
                meter.complete_comparison()
                states[window][index] = 'request_same'
                continue
            _require(result.checks['request_equal'] is False
                     and result.reference_action is not None and result.alternative_action is not None
                     and transitions + 2 <= 2 * MAX_CASE_COMPARISONS)
            meter.complete_comparison()
            if same_request(result.reference_action, result.alternative_action):
                states[window][index] = 'official_request_same'
                continue
            transitions += 2  # Reserve both engine steps before calling the executor.
            meter.reserve_official()
            effect = compare_step(game.replay,step=step,side=side,baseline_request=result.reference_action,
                                  intervention_request=result.alternative_action,guard=guard)
            _require(type(effect) is dict and set(effect) == {'factual_reproduced','request_changed','state_changed',
                'scope','complete_game','future_return_evaluated'} and effect['factual_reproduced'] is True
                and effect['request_changed'] is True and type(effect['state_changed']) is bool
                and effect['scope'] == 'one_official_transition' and effect['complete_game'] is False
                and effect['future_return_evaluated'] is False)
            meter.complete_official()
            if not effect['state_changed']:
                states[window][index] = 'official_next_frame_no_difference_detected'
                continue
            states[window][index] = 'official_next_frame_different'
            candidate = dict(role=role,slot=slot,opponent=opponent,side=side,window=window,
                rank=index,step=step,ordinal=ordinal,input_sha256=ordered_hash(inputs),
                original_choice=record['chosen'],teacher_choice=record['teacher_label'])
    _require(comparisons <= MAX_CASE_COMPARISONS and transitions <= 2*comparisons)
    coverage = dict(role=role,opponent=opponent,side=side,status_by_window=states,
        ranked_out_by_window=outside,source_disagreements_beyond_top4='unknown',
        comparisons=comparisons,official_transitions_reserved=transitions,
        candidate_found=candidate is not None)
    _require(ordered_hash(plan) == plan_sha)
    return PrivateCase(coverage,candidate,plan)
