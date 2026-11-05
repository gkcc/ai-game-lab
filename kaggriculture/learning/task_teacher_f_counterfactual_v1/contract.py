"""Fixed source selection and descriptive statistics; no private I/O."""
import json
import math
import re

BASE_COMMIT = 'e9901942781d074e45f16bde3c290d8c13c6495c'
ENGINE_COMMIT = 'f994f2050d58d84f649329c1e35ce09ada7007c5'
ENGINE_SHA = 'bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e'
COHORTS = {'C01': {'S02': 'F01', 'S03': 'F02'},
           'C02': {'S02': 'F03', 'S03': 'F04'}}
SLOTS = {role: f'T{index:02d}' for index, role in enumerate(('F01','F02','F03','F04'), 1)}
SOURCES = tuple(dict(role=role, opponent=opponent, side=side, policy='baseline')
    for role, opponent in (('F01','market'),('F02','round2'),('F03','market'),('F04','round2'))
    for side in (0, 1))
WINDOWS = ((0,119),(120,239),(240,359),(360,479),(480,599),(600,718))
PHASES = ('prepare_case','control','control_audit','control_verify',
          'intervene','intervene_audit','pair_verify')
MIB, GIB = 1024**2, 1024**3
LIMITS = dict(initial_available_bytes=6*GIB, between_available_bytes=6*GIB,
    running_available_bytes=4*GIB, readiness_samples=3, readiness_interval_seconds=10,
    readiness_timeout_seconds=300, max_rss_bytes=2*GIB, max_commit_bytes=3*GIB,
    job_wall_seconds=900, cleanup_seconds=10, poll_seconds=1,
    screen_wall_seconds=2*60*60, pair_wall_seconds=6*60*60,
    disk_floor_bytes=4*GIB, screen_output_bytes=16*MIB, pair_output_bytes=8*GIB,
    private_result_bytes=16*MIB, max_attempts_per_stage=1,
    max_sources=8, max_screen_comparisons=96, max_screen_controller_branch_reservations=192,
    max_screen_official_transitions=192, max_pairs=8, max_new_games=16,
    max_official_replays=16, max_combined_official_transitions=23200)
PHASE_BYTES = dict(prepare_case=8*MIB, control=160*MIB, control_audit=MIB,
    control_verify=MIB, intervene=160*MIB, intervene_audit=MIB, pair_verify=MIB)
FIELDS = ('own','opponent','margin','score')
STATUSES = frozenset(('empty','unexamined_after_hit','request_same','official_request_same',
    'unknown','official_next_frame_no_difference_detected','official_next_frame_different'))


def require(value):
    if not value:
        raise ValueError('Consumed F counterfactual contract differs')


def finite(value):
    require(type(value) in (int,float) and math.isfinite(value))
    return value


def same(left, right):
    return json.dumps(left,sort_keys=True,allow_nan=False) == json.dumps(right,sort_keys=True,allow_nan=False)


def source_index(source):
    require(type(source) is dict and set(source)=={'role','opponent','side','policy'}
        and type(source['side']) is int and source['side'] in (0,1))
    indices = [i for i, expected in enumerate(SOURCES) if source==expected]
    require(len(indices)==1)
    return indices[0]


def screening_contract():
    return dict(sources=list(SOURCES), slots=SLOTS, windows=[list(x) for x in WINDOWS],
        policy='baseline', ranker='existing_dagger_coordinate_rank_with_fixed_slots',
        candidates_per_window=2, retained_ranked_steps_per_window=4,
        stopping='first_different_official_non_action_next_frame_per_source',
        empty_sources='keep_empty_without_replacement',
        matched_candidates='all_hits_fixed_source_order_separately_committed_before_pairs')


def validate_registration(value):
    require(type(value) is dict and value.get('schema')=='teacher-f-counterfactual-screen-prereg-v1'
        and value.get('base_commit')==BASE_COMMIT
        and value.get('engine_commit')==ENGINE_COMMIT and value.get('engine_sha256')==ENGINE_SHA
        and same(value.get('limits'), LIMITS) and same(value.get('screening'), screening_contract())
        and same(value.get('phase_write_bytes'), PHASE_BYTES)
        and value.get('pair_execution_authorized') is False
        and value.get('training') is False and value.get('gpu') is False
        and value.get('independent_evaluation') is False and value.get('promotion') is False
        and type(value.get('binding_commitment_sha256')) is str
        and re.fullmatch(r'[0-9a-f]{64}',value['binding_commitment_sha256']) is not None)


def screen_projection(results):
    """Only complete fixed-source screening may be summarized publicly."""
    require(type(results) is list and len(results)==len(SOURCES))
    hits, counters, statuses, trace_lines = [], {}, {}, 0
    for index, item in enumerate(results):
        require(type(item) is dict and source_index(item['source'])==index)
        coverage, candidate = item['coverage'], item['candidate']
        require(type(coverage) is dict and type(coverage.get('candidate_found')) is bool
            and coverage['candidate_found']==(candidate is not None))
        matrix = coverage['status_by_window']
        require(type(matrix) is list and len(matrix)==6
            and all(type(row) is list and len(row)==2 for row in matrix))
        flat = [entry for row in matrix for entry in row]
        require(all(type(entry) is str and entry in STATUSES for entry in flat))
        require(flat.count('official_next_frame_different')==int(candidate is not None))
        hit = flat.index('official_next_frame_different') if candidate is not None else len(flat)
        require(all(value in ('empty','unexamined_after_hit') for value in flat[hit+1:])
            and 'unexamined_after_hit' not in flat[:hit])
        comparisons = sum(value not in ('empty','unexamined_after_hit') for value in flat)
        helpers = sum(value in ('official_next_frame_no_difference_detected',
            'official_next_frame_different') for value in flat)
        require(type(coverage.get('source_trace_lines')) is int and coverage['source_trace_lines']==719)
        trace_lines += coverage['source_trace_lines']
        for status in flat:
            require(type(status) is str)
            statuses[status]=statuses.get(status,0)+1
        meter=coverage['meter']
        require(type(meter) is dict and meter['comparisons_begun']==meter['comparisons_completed']
            and meter['official_helpers_begun']==meter['official_helpers_completed']
            and meter['comparisons_reserved']==meter['comparisons_begun']
            and meter['official_transitions_confirmed_completed']==2*meter['official_helpers_completed']
            and meter['official_transitions_reserved']==2*meter['official_helpers_begun']
            and meter['controller_turns_reserved']==2*meter['comparisons_reserved']
            and meter['controller_turns_actual']=='unknown')
        require(meter['comparisons_completed']==comparisons
            and meter['official_helpers_completed']==helpers)
        for key in ('comparisons_reserved','controller_turns_reserved','official_transitions_reserved',
                    'official_transitions_confirmed_completed'):
            require(type(meter[key]) is int and meter[key]>=0)
            counters[key]=counters.get(key,0)+meter[key]
        require(meter['comparisons_reserved']<=12 and meter['official_transitions_reserved']<=24
            and coverage['comparisons']==meter['comparisons_completed']
            and coverage['official_transitions_reserved']==meter['official_transitions_reserved'])
        if candidate is not None:
            require(type(candidate) is dict and all(candidate[k]==item['source'][k]
                for k in ('role','opponent','side')) and type(candidate['window']) is int
                and type(candidate['rank']) is int
                and candidate['window']==hit//2 and candidate['rank']==hit%2)
            hits.append(item['source'])
    require(counters['comparisons_reserved']<=96 and counters['official_transitions_reserved']<=192)
    return dict(schema='teacher-f-counterfactual-screen-public-v1',scope='consumed_development',
        fixed_sources=8,source_trace_lines=trace_lines,candidate_count=len(hits),
        represented_seed_clusters=len({x['role'] for x in hits}),
        represented_opponents=sorted({x['opponent'] for x in hits}),
        represented_sides=sorted({x['side'] for x in hits}),status_counts=statuses,counters=counters,
        actual_screen_controller_turns='unknown',new_complete_games=0,
        future_returns_evaluated=False,training=False,gpu=False,independent_evaluation=False,promotion=False,
        conclusion='requires_separate_pair_freeze' if hits else 'no_measurable_candidate_in_fixed_screen')


def pair_projection(accounts, sources):
    """Case and observed-seed weighted diagnostics, never an admission gate."""
    require(type(accounts) is list and type(sources) is list and len(accounts)==len(sources)<=8)
    indices=[source_index(s) for s in sources]
    require(indices==sorted(set(indices)))
    for account in accounts:
        require(type(account) is dict and set(account)=={'control','intervention','delta'})
        for scope in ('control','intervention','delta'):
            require(type(account[scope]) is dict and set(account[scope])==set(FIELDS))
            for key in FIELDS: finite(account[scope][key])
        for key in FIELDS:
            require(account['delta'][key]==account['intervention'][key]-account['control'][key])
        for arm in ('control','intervention'):
            margin=account[arm]['margin']
            require(account[arm]['score']==(1. if margin>0 else .5 if margin==0 else 0.))
        require(account['delta']['margin']==account['delta']['own']-account['delta']['opponent'])
    if not accounts:
        return dict(pairs=0,represented_seed_clusters=0,case_mean_delta=None,seed_mean_delta=None,
            positive_zero_negative=None,locally_promising_cases=0,promotion=False)
    def means(rows):
        return {key:finite(math.fsum(a['delta'][key] for a in rows)/len(rows)) for key in FIELDS}
    roles=sorted({s['role'] for s in sources})
    grouped=[means([a for a,s in zip(accounts,sources) if s['role']==role]) for role in roles]
    return dict(pairs=len(accounts),represented_seed_clusters=len(roles),case_mean_delta=means(accounts),
        seed_mean_delta={key:finite(math.fsum(row[key] for row in grouped)/len(grouped)) for key in FIELDS},
        positive_zero_negative={key:dict(positive=sum(a['delta'][key]>0 for a in accounts),
            zero=sum(a['delta'][key]==0 for a in accounts),negative=sum(a['delta'][key]<0 for a in accounts)) for key in FIELDS},
        locally_promising_cases=sum(a['delta']['own']>0 and a['delta']['margin']>=0 and a['delta']['score']>=0 for a in accounts),
        promotion=False)
